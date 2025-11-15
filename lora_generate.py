# lora_generate.py
# Colab에서 실행 전 필요한 패키지 설치:
# !pip install diffusers transformers accelerate soundfile librosa torch torchaudio peft

import os
import json
import torch
import soundfile as sf
from tqdm import tqdm
import numpy as np
import argparse
from diffusers import AudioLDMPipeline
from peft import PeftModel

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def save_wav(wav_np, sr, path):
    sf.write(path, wav_np, samplerate=sr)

def generate_text_to_audio_with_lora(model, text, seed=None, duration_sec=6, sr=16000, **kwargs):
    """
    LoRA 적용된 AudioLDM 모델을 사용하여 텍스트로부터 BGM 오디오 생성 (프롬프트 독립적)
    반환: np.ndarray(float32, shape=(n_samples,)) - mono
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    # LoRA: baseline과 동일한 프롬프트 형식 사용 
    # LoRA 학습을 통해 텍스트 내용에서 적절한 BGM 생성
    bgm_prompt = f"instrumental music for {text[:80]}"
    
    print(f"LoRA BGM Prompt: {bgm_prompt}")
        
    # LoRA는 학습을 통해 더 안정적이 될 것으로 기대
    audio = model(
        bgm_prompt,  # baseline과 동일한 프롬프트 형식
        num_inference_steps=20,  # 품질과 다양성 균형
        audio_length_in_s=duration_sec,  # 오디오 길이 (초)
        guidance_scale=4.0,  # baseline과 동일한 값으로 통일
        negative_prompt="noise, static, distortion, glitch, vocals, singing, speech",  # 노이즈 및 보컬 제외
        generator=torch.Generator().manual_seed(seed) if seed is not None else None
    ).audios[0]    # 16kHz로 리샘플링이 필요한 경우를 대비해 확인
    if len(audio.shape) > 1:
        audio = audio[0]  # 첫 번째 채널만 사용 (mono)
    
    return audio

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    # AudioLDM + LoRA 모델 로드
    print("Loading AudioLDM model with LoRA...")
    model = AudioLDMPipeline.from_pretrained(
        "cvssp/audioldm", 
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    
    # LoRA weights 로드 (학습된 LoRA 어댑터 적용)
    if args.lora_weights and os.path.exists(args.lora_weights):
        print(f"Loading LoRA weights from {args.lora_weights}")
        try:
            # PEFT를 사용해서 LoRA weights 로드
            model.unet = PeftModel.from_pretrained(model.unet, args.lora_weights)
            print("✅ LoRA weights loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading LoRA weights: {e}")
            print("Using base model instead.")
    else:
        print("⚠️ Warning: LoRA weights not found or not specified. Using base model.")
    
    print("AudioLDM + LoRA model loaded successfully!")

    # --- 경로 세팅 ---
    test_json = args.test_json  # data/a_data_test.json
    out_test = "output/lora/test"
    ensure_dir(out_test)

    test_data = load_json(test_json)

    # 단일 오디오 생성 모드
    if args.single_index is not None:
        print(f"Generating single LoRA audio for index {args.single_index}")
        
        # 해당 인덱스 찾기 (test 데이터에서만 - 200~232)
        target_item = None
        for item in test_data:
            if item.get("scene_id") == args.single_index:
                target_item = item
                break
        
        if target_item is None:
            print(f"Error: Index {args.single_index} not found in test data (200~232)!")
            print("Available indices in test data:")
            for item in test_data[:10]:  # 처음 10개만 표시
                print(f"  - {item.get('scene_id')}")
            if len(test_data) > 10:
                print(f"  ... and {len(test_data)-10} more")
            return
        
        # 오디오 생성
        idx = target_item.get("scene_id")
        text = target_item.get("text", "").strip()
        
        if not text:
            print(f"Error: Empty text for scene_id {idx}")
            return
        
        print(f"Text: {text[:100]}..." if len(text) > 100 else f"Text: {text}")
        print(f"Generating LoRA audio for scene_id {idx}...")
        
        seed = args.seed + idx if args.seed is not None else None
        wav = generate_text_to_audio_with_lora(model, text, seed=seed, duration_sec=args.duration, sr=args.sr)
        out_path = os.path.join(out_test, f"{idx:03d}_lora.wav")
        save_wav(wav, args.sr, out_path)
        
        print(f"✅ LoRA audio saved: {out_path}")
        return
    
    # 전체 배치 생성 모드 (기존 로직)
    # --- TEST (200~232) - LoRA 적용된 오디오 생성 ---
    print("Generating LoRA test audios...")
    for item in tqdm(test_data, desc="LoRA test audio generation"):
        idx = item.get("scene_id")
        text = item.get("text", "").strip()
        
        if not text:
            print(f"Warning: Empty text for scene_id {idx}, skipping...")
            continue
            
        seed = args.seed + idx if args.seed is not None else None
        wav = generate_text_to_audio_with_lora(model, text, seed=seed, duration_sec=args.duration, sr=args.sr)
        out_path = os.path.join(out_test, f"{idx:03d}_lora.wav")
        save_wav(wav, args.sr, out_path)
        
    print(f"LoRA audio generation completed! Files saved to {out_test}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_json", default="data/a_data_test.json")
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--duration", type=int, default=6)  # 초
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora_weights", type=str, help="LoRA adapter 폴더/파일 경로 (선택사항)")
    parser.add_argument("--single_index", type=int, help="Generate single LoRA audio for specific test index (200~232)")
    args = parser.parse_args()
    main(args)
