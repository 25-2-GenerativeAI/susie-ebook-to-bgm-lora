#!/usr/bin/env python3
"""
lora_generate.py - AudioLDM1 + LoRA 배치 생성 (test 데이터용)

사용법:
  # test 데이터 전체 생성
  python lora_generate.py --batch --test-json data/a_data_test.json
  
  # 단일 index 생성
  python lora_generate.py --single 205 --test-json data/a_data_test.json
"""

import os
import json
import torch
import soundfile as sf
import numpy as np
import argparse
from tqdm import tqdm
from omegaconf import OmegaConf
from peft import PeftModel

try:
    from audioldm.utils import instantiate_from_config
except ImportError:
    print("[ERROR] AudioLDM not found")
    raise


def load_model(config_path, ckpt_path, device):
    """AudioLDM1 기본 모델 로드"""
    print(f"[Model] Loading config: {config_path}")
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model)
    
    print(f"[Model] Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    
    model = model.to(device)
    model.eval()
    print("[Model] ✓ Loaded")
    return model


def load_lora(model, lora_path):
    """LoRA 가중치 로드"""
    if not lora_path or not os.path.exists(lora_path):
        print(f"[LoRA] ⚠️  Not found: {lora_path}")
        return
    
    print(f"[LoRA] Loading: {lora_path}")
    try:
        model.model.diffusion_model = PeftModel.from_pretrained(
            model.model.diffusion_model,
            lora_path,
            is_trainable=False
        )
        print("[LoRA] ✓ Loaded")
    except Exception as e:
        print(f"[LoRA] ❌ {e}")


def generate_single(model, text, seed=None, device='cuda', sr=16000, guidance_scale=7.5, steps=50):
    """단일 텍스트로 오디오 생성"""
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    with torch.no_grad():
        # CLAP 인코딩
        cond = model.get_learned_conditioning([text])
        if cond.dim() == 3:
            cond = cond.mean(dim=1)
        
        uncond = model.get_learned_conditioning([""])
        if uncond.dim() == 3:
            uncond = uncond.mean(dim=1)
        
        # 초기 latent
        C, H, W = 8, 16, 16
        z = torch.randn(1, C, H, W, device=device, dtype=torch.float32)
        
        # Diffusion
        scheduler = model.scheduler
        timesteps = scheduler.timesteps[:steps]
        
        for t in tqdm(timesteps, desc=f"Generating", leave=False):
            pred_cond = model.apply_model(z, t, cond)
            pred_uncond = model.apply_model(z, t, uncond)
            pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
            z = scheduler.step(pred, t, z).prev_sample
        
        # VAE 디코딩
        z_scaled = model.get_first_stage_encoding(z)
        audio = model.decode_first_stage(z_scaled)
        audio = audio.squeeze().cpu().numpy()
        
        # 정규화
        if np.abs(audio).max() > 0:
            audio = audio / np.abs(audio).max() * 0.95
    
    return audio


def main():
    parser = argparse.ArgumentParser(description="AudioLDM1 + LoRA 배치 생성")
    parser.add_argument("--batch", action="store_true", help="배치 모드 (전체 test 데이터)")
    parser.add_argument("--single", type=int, help="단일 index 생성 (예: 205)")
    parser.add_argument("--test-json", default="data/a_data_test.json", help="Test 데이터 JSON")
    parser.add_argument("--config", default="audioldm/configs/audioldm.yaml", help="설정 파일")
    parser.add_argument("--ckpt", default="audioldm_models/audioldm-full.ckpt", help="체크포인트")
    parser.add_argument("--lora", default="lora_weight/final", help="LoRA 가중치")
    parser.add_argument("--output-dir", default="output/lora/test", help="출력 디렉토리")
    parser.add_argument("--sr", type=int, default=16000, help="샘플링 레이트")
    parser.add_argument("--steps", type=int, default=50, help="Diffusion 스텝")
    parser.add_argument("--guidance-scale", type=float, default=7.5, help="Guidance 강도")
    parser.add_argument("--seed", type=int, default=42, help="Base seed")
    
    args = parser.parse_args()
    
    # 배치 또는 단일 중 하나 선택
    if not args.batch and args.single is None:
        print("[ERROR] --batch 또는 --single 옵션이 필요합니다")
        return
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n🎵 AudioLDM1 + LoRA Batch Generation")
    print(f"Device: {device}\n")
    
    # 모델 로드
    model = load_model(args.config, args.ckpt, device)
    load_lora(model, args.lora)
    
    # CRITICAL FIX #3
    if hasattr(model, 'model'):
        model.model.is_lora = True
    
    # 데이터 로드
    with open(args.test_json, 'r') as f:
        test_data = json.load(f)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 단일 생성
    if args.single is not None:
        print(f"[Generate] Single index: {args.single}")
        
        # 해당 인덱스 찾기
        target = None
        for item in test_data:
            if item.get("scene_id") == args.single:
                target = item
                break
        
        if not target:
            print(f"[ERROR] Index {args.single} not found")
            return
        
        text = target.get("text", "").strip()
        print(f"Text: {text[:100]}")
        
        audio = generate_single(
            model, text,
            seed=args.seed + args.single,
            device=device,
            sr=args.sr,
            guidance_scale=args.guidance_scale,
            steps=args.steps
        )
        
        output_file = os.path.join(args.output_dir, f"{args.single:03d}_lora.wav")
        sf.write(output_file, audio, samplerate=args.sr)
        print(f"✅ Saved: {output_file}\n")
    
    # 배치 생성
    else:
        print(f"[Generate] Batch mode ({len(test_data)} items)")
        
        for item in tqdm(test_data, desc="Generating"):
            idx = item.get("scene_id")
            text = item.get("text", "").strip()
            
            if not text:
                print(f"  ⚠️  Skip {idx} (empty text)")
                continue
            
            audio = generate_single(
                model, text,
                seed=args.seed + idx,
                device=device,
                sr=args.sr,
                guidance_scale=args.guidance_scale,
                steps=args.steps
            )
            
            output_file = os.path.join(args.output_dir, f"{idx:03d}_lora.wav")
            sf.write(output_file, audio, samplerate=args.sr)
        
        print(f"\n✅ All generated in: {args.output_dir}")


if __name__ == "__main__":
    main()
