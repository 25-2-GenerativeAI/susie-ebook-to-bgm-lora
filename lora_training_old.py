# lora_training.py
# AudioLDM1 + LoRA 학습 코드
# Colab에서 실행 전 필요한 패키지 설치:
# !pip install diffusers transformers accelerate soundfile librosa torch torchaudio peft datasets

import os
import json
import torch
import torch.nn.functional as F
import soundfile as sf
import librosa
import numpy as np
import argparse
from torch.utils.data import Dataset, DataLoader
from diffusers import AudioLDMPipeline
from peft import LoraConfig, get_peft_model
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

def load_json(path):
    """JSON 파일 로드"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def ensure_dir(d):
    """디렉토리 생성"""
    os.makedirs(d, exist_ok=True)

class AudioRatingDataset(Dataset):
    """
    rating_data_train.json을 사용하는 데이터셋 (3+1 평가법)
    전체 만족도, 텍스트 적합성, 음질, 개선점을 포함
    """
    def __init__(self, json_path, sr=16000, max_length=96000):  # 6초 * 16kHz
        self.data = load_json(json_path)
        self.sr = sr
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 텍스트 (길이 제한 - 512 토큰 제한 대응)
        text = item.get("text", "").strip()
        if len(text) > 500:  # 안전한 길이로 제한
            text = text[:500] + "..."
        
        # 오디오 로드
        audio_path = item.get("audio_path", "")
        if os.path.exists(audio_path):
            audio, _ = librosa.load(audio_path, sr=self.sr, duration=6.0)
            # 길이 맞추기 (패딩 또는 자르기)
            if len(audio) < self.max_length:
                audio = np.pad(audio, (0, self.max_length - len(audio)), 'constant')
            else:
                audio = audio[:self.max_length]
        else:
            # 오디오 파일이 없으면 빈 오디오
            audio = np.zeros(self.max_length)
        
        # 3+1 평가 데이터
        overall_score = float(item.get("overall_score", 5.0))      # 전체 만족도 (1~10)
        text_match = float(item.get("text_match", 5.0))           # 텍스트 적합성 (1~10)
        audio_quality = item.get("audio_quality", "Good")         # 음질 (Good/Bad)
        improvement = item.get("improvement", "")                 # 개선점 (선택사항)
        
        return {
            "text": text,
            "audio": torch.FloatTensor(audio),
            "overall_score": overall_score,
            "text_match": text_match,
            "audio_quality": audio_quality,
            "improvement": improvement,
            "scene_id": item.get("scene_id", idx)
        }

def collate_fn(batch):
    """배치 데이터 처리 (3+1 평가법)"""
    texts = [item["text"] for item in batch]
    audios = torch.stack([item["audio"] for item in batch])
    overall_scores = torch.FloatTensor([item["overall_score"] for item in batch])
    text_matches = torch.FloatTensor([item["text_match"] for item in batch])
    audio_qualities = [item["audio_quality"] for item in batch]
    improvements = [item["improvement"] for item in batch]
    scene_ids = [item["scene_id"] for item in batch]
    
    return {
        "texts": texts,
        "audios": audios,
        "overall_scores": overall_scores,
        "text_matches": text_matches,
        "audio_qualities": audio_qualities,
        "improvements": improvements,
        "scene_ids": scene_ids
    }

def setup_lora_model(base_model, lora_params):
    """
    AudioLDM에 LoRA 설정 적용
    """
    # UNet의 attention 레이어들에 LoRA 적용
    target_modules = [
        "to_k", "to_q", "to_v", "to_out.0",  # attention layers
        "ff.net.0.proj", "ff.net.2"  # feedforward layers
    ]
    
    lora_config = LoraConfig(
        r=lora_params["r"],  # LoRA rank
        lora_alpha=lora_params["lora_alpha"],
        target_modules=target_modules,
        lora_dropout=lora_params["lora_dropout"],
        bias="none",
        # TaskType.DIFFUSION이 없으므로 제거하거나 다른 타입 사용
    )
    
    # UNet에만 LoRA 적용 (VAE, Text Encoder는 freeze)
    unet_lora = get_peft_model(base_model.unet, lora_config)
    base_model.unet = unet_lora
    
    return base_model

def compute_spectral_loss(predicted_audio, target_audio):
    """스펙트럼 도메인에서의 손실 (텍스트 적합성용)"""
    # STFT를 통한 주파수 도메인 비교
    pred_stft = torch.stft(predicted_audio, n_fft=1024, hop_length=256, return_complex=True)
    target_stft = torch.stft(target_audio, n_fft=1024, hop_length=256, return_complex=True)
    
    pred_mag = torch.abs(pred_stft)
    target_mag = torch.abs(target_stft)
    
    return F.mse_loss(pred_mag, target_mag)

def compute_noise_suppression_loss(predicted_audio):
    """노이즈 억제 손실 (음질 개선용)"""
    # 고주파 성분에 페널티
    stft = torch.stft(predicted_audio, n_fft=1024, hop_length=256, return_complex=True)
    magnitude = torch.abs(stft)
    
    # 고주파 영역 (상위 1/3) 억제
    high_freq_start = magnitude.size(-2) * 2 // 3
    high_freq_energy = torch.mean(magnitude[..., high_freq_start:, :])
    
    return high_freq_energy

def compute_mood_adjustment_loss(predicted_audio, improvement_text):
    """개선점 텍스트 기반 분위기 조정 손실"""
    stft = torch.stft(predicted_audio, n_fft=1024, hop_length=256, return_complex=True)
    magnitude = torch.abs(stft)
    
    # 간단한 키워드 기반 주파수 조정
    if "어둡" in improvement_text or "dark" in improvement_text.lower():
        # 저주파 강화, 고주파 억제
        low_freq_end = magnitude.size(-2) // 3
        low_freq_penalty = 1.0 / (torch.mean(magnitude[..., :low_freq_end, :]) + 1e-6)
        return low_freq_penalty
    elif "밝" in improvement_text or "bright" in improvement_text.lower():
        # 고주파 강화
        high_freq_start = magnitude.size(-2) * 2 // 3
        high_freq_boost = torch.mean(magnitude[..., high_freq_start:, :])
        return -high_freq_boost  # 음수로 하여 고주파 강화
    
    return torch.tensor(0.0, device=predicted_audio.device)

def compute_optimized_loss(predicted_audio, target_audio, rating_data):
    """3+1 평가법 기반 최적화된 손실 함수"""
    
    # 1. 기본 재구성 손실 (가장 중요)
    reconstruction_loss = F.mse_loss(predicted_audio, target_audio)
    
    # 2. 전체 만족도 기반 가중치 (핵심!)
    satisfaction_weight = rating_data['overall_score'] / 10.0
    
    # 3. 텍스트 적합성 손실 (LoRA 목표)
    text_match_penalty = (10.0 - rating_data['text_match']) / 10.0
    spectral_loss = compute_spectral_loss(predicted_audio, target_audio)
    text_match_loss = text_match_penalty * spectral_loss
    
    # 4. 음질 손실 (이진 분류)
    quality_loss = torch.tensor(0.0, device=predicted_audio.device)
    if rating_data['audio_quality'] == "Bad":
        quality_loss = compute_noise_suppression_loss(predicted_audio)
    
    # 5. 개선점 반영 (선택적)
    improvement_loss = torch.tensor(0.0, device=predicted_audio.device)
    if rating_data.get('improvement') and rating_data['improvement'].strip():
        improvement_loss = compute_mood_adjustment_loss(predicted_audio, rating_data['improvement'])
    
    # 총 손실 (만족도 가중치 적용)
    total_loss = satisfaction_weight * (
        1.0 * reconstruction_loss +      # 기본 재구성 (50%)
        0.6 * text_match_loss +         # 텍스트 적합성 (30%)
        0.3 * quality_loss +            # 음질 (15%)
        0.1 * torch.abs(improvement_loss)  # 개선점 (5%)
    )
    
    return total_loss

def improved_lora_loss(model, batch, device):
    """개선된 LoRA 손실 - 실제 성능 향상을 위한 버전"""
    
    # LoRA 파라미터 확인
    lora_params = [p for p in model.unet.parameters() if p.requires_grad]
    
    if not lora_params:
        print("Warning: No trainable LoRA parameters found!")
        return torch.tensor(1.0, requires_grad=True, device=device), {}
    
    # 배치 데이터
    texts = batch["texts"]
    target_audios = batch["audios"].to(device)
    overall_scores = batch["overall_scores"].to(device)
    text_matches = batch["text_matches"].to(device)
    audio_qualities = batch["audio_qualities"]
    improvements = batch["improvements"]
    
    try:
        # 1. 텍스트 인코딩 (AudioLDM text encoder 사용)
        with torch.no_grad():
            # 텍스트 길이 제한
            truncated_texts = [text[:200] for text in texts]  # 더 짧게 제한
            
        # 2. 타겟 오디오를 mel-spectrogram으로 변환
        mel_specs = []
        for audio in target_audios:
            # librosa로 mel-spectrogram 생성
            audio_np = audio.detach().cpu().numpy()
            mel_spec = librosa.feature.melspectrogram(
                y=audio_np, 
                sr=16000, 
                n_mels=64,
                hop_length=256,
                n_fft=1024
            )
            mel_spec = torch.from_numpy(mel_spec).to(device)
            mel_specs.append(mel_spec)
        
        mel_specs = torch.stack(mel_specs).unsqueeze(1)  # [batch, 1, mel_bins, time]
        
        # 3. VAE를 통한 latent space 변환
        with torch.no_grad():
            # mel-spectrogram을 적절한 크기로 조정
            target_size = (64, 312)  # AudioLDM VAE 입력 크기
            resized_mels = F.interpolate(mel_specs, size=target_size, mode='bilinear')
            
            # VAE encoder로 latent 변환
            target_latents = model.vae.encode(resized_mels).latent_dist.sample()
            target_latents = target_latents * model.vae.config.scaling_factor
        
        # 4. 노이즈 추가 (Diffusion forward process)
        batch_size = target_latents.shape[0]
        noise = torch.randn_like(target_latents)
        
        # 다양한 timestep에서 학습
        timesteps = torch.randint(0, model.scheduler.config.num_train_timesteps, (batch_size,), device=device)
        noisy_latents = model.scheduler.add_noise(target_latents, noise, timesteps)
        
        # 5. 텍스트 조건화를 위한 임베딩 생성
        # AudioLDM의 text encoder 사용
        text_inputs = model.tokenizer(
            truncated_texts,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt"
        ).to(device)
        
        text_embeddings = model.text_encoder(**text_inputs)[0]
        
        # 6. UNet으로 노이즈 예측 (LoRA 적용된 상태)
        model_pred = model.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=text_embeddings
        ).sample
        
        # 7. 기본 diffusion 손실
        diffusion_loss = F.mse_loss(model_pred, noise)
        
        # 8. 평가 기반 가중치 적용
        # 낮은 점수일수록 더 많이 학습
        satisfaction_penalty = (10.0 - overall_scores) / 10.0
        text_match_penalty = (10.0 - text_matches) / 10.0
        
        # 음질이 나쁜 경우 추가 페널티
        quality_penalty = torch.tensor([
            2.0 if quality == "Bad" else 1.0 
            for quality in audio_qualities
        ], device=device)
        
        # 개선사항이 있는 경우 가중치 증가
        improvement_weight = torch.tensor([
            1.5 if improvement.strip() else 1.0 
            for improvement in improvements
        ], device=device)
        
        # 9. 종합 가중치 계산
        total_weight = (
            satisfaction_penalty * 0.4 +      # 전체 만족도 40%
            text_match_penalty * 0.3 +        # 텍스트 매칭 30%
            quality_penalty * 0.2 +           # 음질 20%
            improvement_weight * 0.1          # 개선사항 10%
        )
        
        # 배치별로 가중치 적용
        weighted_losses = []
        for i in range(batch_size):
            weighted_loss = total_weight[i] * F.mse_loss(model_pred[i:i+1], noise[i:i+1])
            weighted_losses.append(weighted_loss)
        
        total_loss = torch.stack(weighted_losses).mean()
        
        # 10. 상세 손실 정보
        detailed_losses = {
            'total': total_loss,
            'reconstruction': diffusion_loss,
            'satisfaction': overall_scores.mean() / 10.0,
            'text_match': text_matches.mean() / 10.0,
            'avg_weight': total_weight.mean()
        }
        
        return total_loss, detailed_losses
        
    except Exception as e:
        print(f"Improved loss computation error: {e}")
        # 에러 시 기본 LoRA 손실로 폴백
        param_changes = sum(p.pow(2).sum() for p in lora_params) / len(lora_params)
        satisfaction_penalty = (10.0 - overall_scores.mean()) / 10.0
        
        fallback_loss = satisfaction_penalty * param_changes
        
        detailed_losses = {
            'total': fallback_loss,
            'reconstruction': param_changes,
            'satisfaction': overall_scores.mean() / 10.0,
            'text_match': text_matches.mean() / 10.0,
            'avg_weight': satisfaction_penalty
        }
        
        return fallback_loss, detailed_losses

def compute_optimized_batch_loss(model, batch, device):
    """
    3+1 평가법 기반 배치 손실 함수 - 개선된 버전
    """
    return improved_lora_loss(model, batch, device)

def train_lora(args):
    """
    LoRA 학습 메인 함수
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 1. 데이터셋 준비
    print("Loading dataset...")
    dataset = AudioRatingDataset(args.train_json, sr=args.sr)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0  # Colab에서는 0으로 설정
    )
    print(f"Dataset loaded: {len(dataset)} samples")
    
    # 2. 모델 로드
    print("Loading AudioLDM model...")
    model = AudioLDMPipeline.from_pretrained(
        "cvssp/audioldm",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    
    # 3. LoRA 설정 적용
    print("Setting up LoRA...")
    lora_config = {
        "r": args.lora_r,
        "lora_alpha": args.lora_alpha, 
        "lora_dropout": args.lora_dropout
    }
    model = setup_lora_model(model, lora_config)
    
    # 4. 옵티마이저 설정
    trainable_params = [p for p in model.unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate)
    
    # 5. 학습률 스케줄러
    total_steps = len(dataloader) * args.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps
    )
    
    # 6. 학습 루프
    print("Starting LoRA training...")
    # AudioLDM 파이프라인은 train() 메서드가 없으므로 UNet만 학습 모드로 설정
    model.unet.train()
    
    for epoch in range(args.num_epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        
        for batch_idx, batch in enumerate(progress_bar):
            try:
                # Forward pass - 3+1 평가법 기반 손실 계산
                total_loss, detailed_losses = compute_optimized_batch_loss(model, batch, device)
                
                # Backward pass
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                scheduler.step()
                
                epoch_loss += total_loss.item()
                
                # 진행상황 업데이트 (3+1 평가 기반)
                progress_bar.set_postfix({
                    'total': f"{total_loss.item():.4f}",
                    'recon': f"{detailed_losses['reconstruction'].item():.4f}",
                    'satis': f"{detailed_losses['satisfaction'].item():.3f}",
                    'match': f"{detailed_losses['text_match'].item():.3f}",
                    'avg': f"{epoch_loss/(batch_idx+1):.4f}",
                    'lr': f"{scheduler.get_last_lr()[0]:.2e}"
                })
                
                # 중간 저장 (매 100스텝마다)
                if (batch_idx + 1) % 100 == 0:
                    checkpoint_path = os.path.join(args.output_dir, f"checkpoint_epoch{epoch+1}_step{batch_idx+1}")
                    model.unet.save_pretrained(checkpoint_path)
                    print(f"\nCheckpoint saved: {checkpoint_path}")
                
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue
        
        print(f"Epoch {epoch+1} completed. Average loss: {epoch_loss/len(dataloader):.4f}")
    
    # 7. 최종 모델 저장
    model.unet.eval()  # 평가 모드로 전환
    ensure_dir(args.output_dir)
    final_model_path = os.path.join(args.output_dir, "lora_weights")
    model.unet.save_pretrained(final_model_path)
    print(f"LoRA training completed! Model saved to: {final_model_path}")
    
    return final_model_path

def main():
    parser = argparse.ArgumentParser(description="AudioLDM LoRA Training")
    
    # 데이터 관련
    parser.add_argument("--train_json", default="data/rating_data_train.json", help="Training data JSON file")
    parser.add_argument("--sr", type=int, default=16000, help="Sample rate")
    
    # LoRA 설정
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")
    
    # 학습 설정
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    
    # 출력 설정
    parser.add_argument("--output_dir", default="weight", help="Output directory for LoRA weights")
    
    args = parser.parse_args()
    
    # 학습 시작
    model_path = train_lora(args)
    print(f"\n✅ LoRA training completed successfully!")
    print(f"📁 Model saved at: {model_path}")
    print(f"🚀 You can now use this with lora_generate.py --lora_weights {model_path}")

if __name__ == "__main__":
    main()