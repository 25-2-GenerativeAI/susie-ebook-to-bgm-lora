# baseline_generate.py
# Colab에서 실행 전 필요한 패키지 설치:
# !pip install diffusers transformers accelerate soundfile librosa torch torchaudio

import os
import json
import torch
import soundfile as sf
from tqdm import tqdm
import numpy as np
import argparse
from diffusers import AudioLDMPipeline

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def save_wav(wav_np, sr, path):
    sf.write(path, wav_np, samplerate=sr)

def generate_text_to_audio(model, text, seed=None, duration_sec=6, sr=16000, **kwargs):
    """
    AudioLDM 모델을 사용하여 텍스트로부터 BGM 오디오 생성 (Baseline - 프롬프트 의존적)
    반환: np.ndarray(float32, shape=(n_samples,)) - mono
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    # Baseline: 텍스트 내용 반영하되 안전한 프롬프트 템플릿 사용
    bgm_prompt = f"instrumental music for {text[:80]}"
    
    print(f"Baseline BGM Prompt: {bgm_prompt}")
        
    # 다양성 확보하면서도 안전한 파라미터
    audio = model(
        bgm_prompt, 
        num_inference_steps=20,  # 품질과 다양성 균형
        audio_length_in_s=duration_sec,  # 오디오 길이 (초)
        guidance_scale=4.0,  # 텍스트 반영도 높이면서도 안정성 유지
        negative_prompt="noise, static, distortion, glitch, vocals, singing, speech",  # 노이즈 및 보컬 제외
        generator=torch.Generator().manual_seed(seed) if seed is not None else None
    ).audios[0]    # 16kHz로 리샘플링이 필요한 경우를 대비해 확인
    if len(audio.shape) > 1:
        audio = audio[0]  # 첫 번째 채널만 사용 (mono)
    
    return audio

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    # AudioLDM 모델 로드
    print("Loading AudioLDM model...")
    model = AudioLDMPipeline.from_pretrained(
        "cvssp/audioldm", 
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    print("AudioLDM model loaded successfully!")

    # --- 경로 세팅 ---
    train_json = args.train_json  # data/a_data_train.json
    test_json = args.test_json    # data/a_data_test.json

    out_train = "output/baseline/train"
    out_test = "output/baseline/test"
    ensure_dir(out_train)
    ensure_dir(out_test)

    train_data = load_json(train_json)
    test_data = load_json(test_json)

    # 단일 오디오 생성 모드
    if args.single_index is not None:
        print(f"Generating single audio for index {args.single_index}")
        
        # 해당 인덱스 찾기
        target_item = None
        is_test = False
        
        # train 데이터에서 찾기 (0~199)
        for item in train_data:
            if item.get("scene_id") == args.single_index:
                target_item = item
                output_dir = out_train
                break
        
        # test 데이터에서 찾기 (200~232)
        if target_item is None:
            for item in test_data:
                if item.get("scene_id") == args.single_index:
                    target_item = item
                    output_dir = out_test
                    is_test = True
                    break
        
        if target_item is None:
            print(f"Error: Index {args.single_index} not found in data!")
            return
        
        # 오디오 생성
        idx = target_item.get("scene_id")
        text = target_item.get("text", "").strip()
        
        if not text:
            print(f"Error: Empty text for scene_id {idx}")
            return
        
        print(f"Text: {text[:100]}..." if len(text) > 100 else f"Text: {text}")
        print(f"Generating audio for scene_id {idx}...")
        
        seed = args.seed + idx if args.seed is not None else None
        wav = generate_text_to_audio(model, text, seed=seed, duration_sec=args.duration, sr=args.sr)
        out_path = os.path.join(output_dir, f"{idx:03d}_baseline.wav")
        save_wav(wav, args.sr, out_path)
        
        print(f"✅ Audio saved: {out_path}")
        print(f"📁 Type: {'Test' if is_test else 'Train'} data")
        return
    
    # 전체 배치 생성 모드 (기존 로직)
    # --- TRAIN (0~199) ---
    print("Generating baseline train audios...")
    for item in tqdm(train_data, desc="Train audio generation"):
        idx = item.get("scene_id")
        text = item.get("text", "").strip()
        
        if not text:
            print(f"Warning: Empty text for scene_id {idx}, skipping...")
            continue
            
        # 생성 파라미터
        seed = args.seed + idx if args.seed is not None else None
        wav = generate_text_to_audio(model, text, seed=seed, duration_sec=args.duration, sr=args.sr)
        out_path = os.path.join(out_train, f"{idx:03d}_baseline.wav")
        save_wav(wav, args.sr, out_path)

    # --- TEST (200~232) ---
    print("Generating baseline test audios...")
    for item in tqdm(test_data, desc="Test audio generation"):
        idx = item.get("scene_id")
        text = item.get("text", "").strip()
        
        if not text:
            print(f"Warning: Empty text for scene_id {idx}, skipping...")
            continue
            
        seed = args.seed + idx if args.seed is not None else None
        wav = generate_text_to_audio(model, text, seed=seed, duration_sec=args.duration, sr=args.sr)
        out_path = os.path.join(out_test, f"{idx:03d}_baseline.wav")
        save_wav(wav, args.sr, out_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_json", default="data/a_data_train.json")
    parser.add_argument("--test_json", default="data/a_data_test.json")
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--duration", type=int, default=6)  # 초
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--single_index", type=int, help="Generate single audio for specific index (0~232)")
    args = parser.parse_args()
    main(args)
