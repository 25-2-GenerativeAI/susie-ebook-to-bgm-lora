#!/usr/bin/env python3
"""
AudioLDM1 LoRA 훈련 스크립트 (Colab 최적화, yaml 불필요)

사용법:
  python train_lora.py --json-path data/rating_data_train.json --epochs 3 --batch-size 2
"""

import os
import sys
import torch
import torch.nn.functional as F
import torchaudio.functional as tF
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

from lora_unet import apply_lora_to_unet
from dataset import AudioRatingDataset
from torchlibrosa.stft import Spectrogram, LogmelFilterBank


def collate_fn_stable(batch):
    """
    Batch 처리 함수
    텍스트, 오디오, 점수를 반환
    """
    texts = []
    audios = []
    text_matches = []
    
    for item in batch:
        # 텍스트는 반드시 string이어야 함
        text = item["text"]
        if not isinstance(text, str):
            text = str(text)
        texts.append(text)
        
        audios.append(item["audio"])
        text_matches.append(item["text_match"])
    
    return {
        "text": texts,  # string list for text conditioning
        "audio": torch.stack(audios),
        "text_match": torch.FloatTensor(text_matches),
    }


# AudioLDM import (from cloned repo)
sys.path.insert(0, "./AudioLDM")
try:
    from audioldm.utils import instantiate_from_config, default_audioldm_config
    AUDIOLDM_AVAILABLE = True
except ImportError as e:
    print("[ERROR] AudioLDM not found in ./AudioLDM")
    print(f"Error: {e}")
    AUDIOLDM_AVAILABLE = False


def train_lora(
    json_path="data/rating_data_train.json",
    ckpt_path="audioldm_models/audioldm-full.ckpt",
    batch_size=2,
    epochs=3,
    lr=1e-4,
    lora_rank=8,
    output_dir="lora_weight",
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    """AudioLDM1 + LoRA 파인튜닝 훈련 (3개 critical fix 적용)"""
    
    if not AUDIOLDM_AVAILABLE:
        print("[ERROR] AudioLDM not available")
        return

    if not os.path.exists(ckpt_path):
        print(f"[ERROR] Checkpoint not found: {ckpt_path}")
        print("Download with:")
        print("!mkdir -p audioldm_models")
        print("!wget -O audioldm_models/audioldm-full.ckpt <checkpoint_url>")
        return

    print("=" * 70)
    print("🎵 AudioLDM1 LoRA Fine-tuning")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Batch: {batch_size}, Epochs: {epochs}, LR: {lr}")
    print()

    # 1. 모델 로드 (yaml 없이 default config 사용)
    print("[Model] Loading AudioLDM1...")
    cfg = default_audioldm_config()
    
    # CRITICAL FIX: CLAP encoder를 text mode로 설정
    # AudioLDM1은 텍스트 조건을 사용합니다
    cfg["model"]["params"]["cond_stage_config"]["params"]["embed_mode"] = "text"
    
    model = instantiate_from_config(cfg["model"])
    
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    
    model = model.to(device)
    
    # CRITICAL FIX #5: Keep entire model in train() mode for checkpoint backward compatibility
    # AudioLDM1's CheckpointFunction.backward() ONLY works in training mode
    # Even though VAE/CLAP have requires_grad=False, model must be in train() for gradient flow
    # Only LoRA-injected parameters will actually be trained
    model.train()
    
    # Disable gradient checkpointing for stable backward pass
    if hasattr(model, 'model'):
        if hasattr(model.model, 'diffusion_model'):
            model.model.diffusion_model.use_checkpoint = False
    
    print("[Model] ✓ Loaded (entire model in training mode)")
    print()

    # 2. LoRA 적용
    print(f"[LoRA] Preparing LoRA...")
    unet = model.model.diffusion_model
    unet = apply_lora_to_unet(unet, r=lora_rank, alpha=lora_rank * 2)
    model.model.diffusion_model = unet
    
    # CRITICAL FIX #3: 호환성 플래그
    model.model.is_lora = True
    print("[LoRA] ✓ Applied")
    print()

    # 3. 옵티마이저
    trainable_params = [p for p in unet.parameters() if p.requires_grad]
    if not trainable_params:
        print("[ERROR] No trainable parameters!")
        return
    
    optimizer = torch.optim.AdamW(trainable_params, lr=lr)
    print(f"[Optimizer] ✓ {len(trainable_params)} parameter groups")
    print()

    # 4. 데이터셋
    print("[Dataset] Loading...")
    dataset = AudioRatingDataset(json_path, sr=16000, max_length=96000)
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0,
        collate_fn=collate_fn_stable  # ✅ 안정화된 collate_fn 사용
    )
    print(f"[Dataset] ✓ {len(loader)} batches")
    print()

    # 5. MEL-SPECTROGRAM EXTRACTORS (Pre-initialize for efficiency)
    print("[Audio] Preparing mel-spectrogram extractors...")
    spectrogram_extractor = Spectrogram(n_fft=2048, hop_length=512, win_length=2048, center=True, pad_mode='reflect', freeze_parameters=True).to(device)
    logmel_extractor = LogmelFilterBank(sr=48000, n_fft=2048, n_mels=64, fmin=50, fmax=14000, is_log=True, ref=1.0, amin=1e-10, top_db=None).to(device)
    print("[Audio] ✓ Ready")
    print()

    # 5. 훈련 루프 (TEXT CONDITIONING)
    print("[Training] Starting...")
    print("=" * 70)

    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch_idx, batch in enumerate(pbar):
            wav = batch["audio"].to(device)
            texts = batch["text"]  # TEXT conditioning
            text_match = batch["text_match"].to(device)

            # Step 1: TEXT CONDITIONING (CLAP text encoder) - no grad
            with torch.no_grad():
                cond = model.get_learned_conditioning(texts)
                
                # CRITICAL FIX #2: 3D → 2D (temporal pooling)
                if cond.dim() == 3:
                    cond = cond.mean(dim=1)

            # Step 2: MEL-SPECTROGRAM (AudioLDM의 VAE 입력)
            # Audio preprocessing: resample to 48kHz (CLAP's standard)
            wav_48k = tF.resample(wav, orig_freq=16000, new_freq=48000)
            
            # Generate mel-spectrogram using pre-initialized extractors
            with torch.no_grad():
                spec = spectrogram_extractor(wav_48k)  # [B, 1, freq, time]
                mel = logmel_extractor(spec)  # [B, 1, mel_bins, time]
                
                # CRITICAL FIX #1: Mel shape 정규화 → [B, 1, 64, 256]
                if mel.shape[-1] != 256:
                    mel = F.interpolate(mel, size=(64, 256), mode='bilinear', align_corners=False)

                # Step 3: VAE 인코딩
                encoder_posterior = model.encode_first_stage(mel)
                latent = model.get_first_stage_encoding(encoder_posterior)

            # Step 4: Diffusion 스텝
            t = torch.randint(0, model.num_timesteps, (wav.shape[0],), device=device)
            noise = torch.randn_like(latent)
            noisy_latent = model.q_sample(latent, t, noise)

            # Step 5: UNet 예측 (TEXT 조건) - WITH gradient tracking
            # CRITICAL FIX: gradient 흐름을 명시적으로 활성화
            with torch.set_grad_enabled(True):
                pred_noise = model.apply_model(noisy_latent, t, cond)
            
            # 디버깅: gradient 추적 확인 (첫 배치에서만)
            if batch_idx == 0 and epoch == 0:
                print(f"\n[DEBUG] pred_noise.requires_grad = {pred_noise.requires_grad}")
                print(f"[DEBUG] pred_noise.shape = {pred_noise.shape}")

            # Step 6: Loss 계산
            loss = F.mse_loss(pred_noise, noise)
            
            # Rating 기반 가중치 [1.0, 1.5]
            rating_weight = 1.0 + (10.0 - text_match) / 20.0
            loss = (loss * rating_weight).mean()

            # Step 7: Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = epoch_loss / max(num_batches, 1)
        print(f"✓ Epoch {epoch+1} | Loss: {avg_loss:.4f}")

    print("=" * 70)
    print("[Save] Saving LoRA...")
    os.makedirs(output_dir, exist_ok=True)

    lora_path = os.path.join(output_dir, "final")
    if hasattr(unet, 'save_pretrained'):
        unet.save_pretrained(lora_path)
    else:
        torch.save(unet.state_dict(), os.path.join(output_dir, "adapter_model.bin"))
    
    print(f"✓ Saved to {lora_path}")
    print("\n🎉 Training complete!")


def main():
    parser = argparse.ArgumentParser(description="AudioLDM1 LoRA 훈련")
    parser.add_argument("--json-path", default="data/rating_data_train.json", help="훈련 데이터 JSON")
    parser.add_argument("--ckpt-path", default="audioldm_models/audioldm-full.ckpt", help="체크포인트")
    parser.add_argument("--batch-size", type=int, default=2, help="배치 크기")
    parser.add_argument("--epochs", type=int, default=3, help="에포크 수")
    parser.add_argument("--lr", type=float, default=1e-4, help="학습률")
    parser.add_argument("--lora-rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--output-dir", default="lora_weight", help="출력 디렉토리")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()
    train_lora(**vars(args))


if __name__ == "__main__":
    main()
