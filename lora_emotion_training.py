# lora_emotion_training.py
# AudioLDM1 + LoRA 학습 코드 (감정 태깅 포함)
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

class AudioRatingEmotionDataset(Dataset):
    """
    z_rating_data_train.json을 사용하는 데이터셋 (3+1 평가법 + 감정 태깅)
    전체 만족도, 텍스트 적합성, 음질, 개선점, 감정 정보를 포함
    """
    def __init__(self, json_path, sr=16000, max_length=96000):  # 6초 * 16kHz
        self.data = load_json(json_path)
        self.sr = sr
        self.max_length = max_length
        
        # 감정별 가중치 (학습 시 감정에 따른 차별화)
        self.emotion_weights = {
            "joy": 1.2,      # 밝은 음악 강화
            "sadness": 1.1,  # 슬픈 음악 강화
            "anger": 1.0,    # 강렬한 음악 강화
            "fear": 0.9,     # 어두운 음악 강화
            "surprise": 1.0, # 변화가 있는 음악
            "neutral": 0.8,  # 기본 음악
            "love": 1.1,     # 로맨틱한 음악 강화
            "disgust": 0.7   # 불쾌감 억제
        }
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 텍스트 (길이 제한 - 512 토큰 제한 대응)
        text = item.get("text", "").strip()
        if len(text) > 500:  # 안전한 길이로 제한
            text = text[:500] + "..."
        
        # 감정 정보 추가
        dominant_emotion = item.get("dominant_emotion", "neutral")
        emotion_score = item.get("score", 0.5)  # 감정 신뢰도
        emotion_weight = self.emotion_weights.get(dominant_emotion, 1.0)
        
        # 오디오 로드
        audio_path = item.get("audio_path", "")
        audio = None
        if os.path.exists(audio_path):
            try:
                audio, _ = librosa.load(audio_path, sr=self.sr)
                # 길이 맞추기
                if len(audio) > self.max_length:
                    audio = audio[:self.max_length]
                elif len(audio) < self.max_length:
                    audio = np.pad(audio, (0, self.max_length - len(audio)), mode='constant')
            except Exception as e:
                print(f"Audio load error: {audio_path}, {e}")
                audio = np.zeros(self.max_length)
        else:
            audio = np.zeros(self.max_length)  # 기본값
        
        # 평가 점수 (3+1 평가법)
        overall_score = float(item.get("overall_score", 0))  # 전체 만족도 (1~10)
        text_match = float(item.get("text_match", 0))        # 텍스트 적합성 (1~10)
        audio_quality = item.get("audio_quality", "Bad")     # 음질 (Good/Bad)
        improvement = item.get("improvement", "")            # 개선점
        
        # 음질을 수치로 변환
        quality_score = 1.0 if audio_quality == "Good" else 0.0
        
        return {
            "text": text,
            "audio": torch.FloatTensor(audio),
            "overall_score": overall_score,
            "text_match": text_match,
            "quality_score": quality_score,
            "improvement": improvement,
            "dominant_emotion": dominant_emotion,    # 감정 정보
            "emotion_score": emotion_score,          # 감정 신뢰도
            "emotion_weight": emotion_weight,        # 감정별 가중치
            "scene_id": item.get("scene_id", 0)
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

def compute_emotion_aware_loss(predicted_audio, dominant_emotion, emotion_score):
    """
    감정 인식 기반 손실 함수 (NEW!)
    감정에 맞는 주파수 특성을 강화하는 손실
    """
    stft = torch.stft(predicted_audio, n_fft=1024, hop_length=256, return_complex=True)
    magnitude = torch.abs(stft)
    
    # 주파수 대역별 에너지
    low_freq = torch.mean(magnitude[..., :magnitude.size(-2)//4, :])    # 저주파 (0-25%)
    mid_freq = torch.mean(magnitude[..., magnitude.size(-2)//4:magnitude.size(-2)*3//4, :])  # 중주파 (25-75%)
    high_freq = torch.mean(magnitude[..., magnitude.size(-2)*3//4:, :]) # 고주파 (75-100%)
    
    emotion_loss = 0.0
    
    if dominant_emotion == "joy":
        # 기쁨: 밝고 경쾌한 중고주파 강화
        emotion_loss = -0.5 * mid_freq - 0.3 * high_freq + 0.2 * low_freq
    elif dominant_emotion == "sadness":
        # 슬픔: 저주파 강화, 고주파 억제
        emotion_loss = -0.6 * low_freq + 0.3 * high_freq
    elif dominant_emotion == "anger":
        # 분노: 강렬한 모든 주파수 강화
        emotion_loss = -0.4 * (low_freq + mid_freq + high_freq)
    elif dominant_emotion == "fear":
        # 공포: 어둡고 불안한 저주파 + 날카로운 고주파
        emotion_loss = -0.3 * low_freq - 0.2 * high_freq + 0.5 * mid_freq
    elif dominant_emotion == "love":
        # 사랑: 따뜻한 중저주파 강화
        emotion_loss = -0.5 * low_freq - 0.4 * mid_freq + 0.1 * high_freq
    elif dominant_emotion == "neutral":
        # 중성: 균형잡힌 주파수 분포
        emotion_loss = 0.1 * torch.abs(low_freq - mid_freq) + 0.1 * torch.abs(mid_freq - high_freq)
    else:
        # 기타: 기본 처리
        emotion_loss = 0.0
    
    # 감정 신뢰도로 가중치 적용
    return emotion_loss * emotion_score

def compute_mood_adjustment_loss(predicted_audio, improvement_text):
    """개선점 텍스트 기반 분위기 조정 손실"""
    if not improvement_text or improvement_text.strip() == "":
        return torch.tensor(0.0, device=predicted_audio.device)
    
    # 개선점에 따른 조정
    improvement_lower = improvement_text.lower()
    
    stft = torch.stft(predicted_audio, n_fft=1024, hop_length=256, return_complex=True)
    magnitude = torch.abs(stft)
    
    penalty = 0.0
    
    if "목소리" in improvement_text or "vocal" in improvement_lower or "speech" in improvement_lower:
        # 음성 성분 억제 (중주파 대역)
        vocal_freq_range = magnitude[..., magnitude.size(-2)//4:magnitude.size(-2)*3//4, :]
        penalty += torch.mean(vocal_freq_range) * 0.5
    
    if "시끄" in improvement_text or "noise" in improvement_lower or "distortion" in improvement_lower:
        # 노이즈 억제 (고주파)
        noise_freq_range = magnitude[..., magnitude.size(-2)*3//4:, :]
        penalty += torch.mean(noise_freq_range) * 0.3
        
    if "단조" in improvement_text or "boring" in improvement_lower:
        # 변화 부족 개선 (주파수 다양성 증가)
        freq_variance = torch.var(magnitude, dim=-2)
        penalty -= torch.mean(freq_variance) * 0.2
    
    return penalty

def improved_lora_loss_with_emotion(model, batch, device):
    """
    감정 정보를 활용한 개선된 LoRA diffusion 손실 함수
    """
    texts = batch["text"]
    audios = batch["audio"].to(device)
    overall_scores = batch["overall_score"].to(device)
    text_matches = batch["text_match"].to(device)
    quality_scores = batch["quality_score"].to(device)
    improvements = batch["improvement"]
    dominant_emotions = batch["dominant_emotion"]      # 감정 정보
    emotion_scores = batch["emotion_score"].to(device) # 감정 신뢰도
    emotion_weights = batch["emotion_weight"].to(device) # 감정별 가중치
    
    batch_size = len(texts)
    
    # 텍스트 길이 제한 (토큰 한계 대응)
    truncated_texts = [f"instrumental music for {text[:80]}" for text in texts]
    
    # 1. 멜 스펙트로그램 변환 (16kHz → 16kHz, 1024 mel bins)
    target_mels = []
    for audio in audios:
        # librosa를 사용한 멜 스펙트로그램 생성 (AudioLDM 호환)
        mel_spec = librosa.feature.melspectrogram(
            y=audio.cpu().numpy(),
            sr=16000,
            n_fft=1024,
            hop_length=160,
            n_mels=64,
            fmin=0,
            fmax=8000
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        target_mels.append(torch.FloatTensor(mel_spec_db))
    
    # 2. 배치로 스택 (패딩으로 크기 맞추기)
    max_frames = max(mel.shape[1] for mel in target_mels)
    padded_mels = []
    for mel in target_mels:
        if mel.shape[1] < max_frames:
            pad_width = max_frames - mel.shape[1]
            mel = F.pad(mel, (0, pad_width), mode='constant', value=-80.0)  # 조용한 부분으로 패딩
        padded_mels.append(mel)
    
    mel_batch = torch.stack(padded_mels).unsqueeze(1).to(device)  # [B, 1, 64, T]
    
    # 데이터 타입을 모델과 일치시키기
    if model.vae.dtype == torch.float16:
        mel_batch = mel_batch.half()
    else:
        mel_batch = mel_batch.float()
    
    # 3. VAE 인코딩 (AudioLDM의 멜 → 잠재공간)
    with torch.no_grad():
        # 크기 조정이 필요한 경우
        if mel_batch.shape[-1] % 4 != 0:
            target_width = (mel_batch.shape[-1] // 4) * 4
            if target_width == 0:
                target_width = 4
            mel_batch = F.interpolate(mel_batch, size=(64, target_width), mode='bilinear', align_corners=False)
        
        target_latents = model.vae.encode(mel_batch).latent_dist.sample()
        target_latents = target_latents * model.vae.config.scaling_factor
    
    # 4. 노이즈 추가 (Diffusion forward process)
    noise = torch.randn_like(target_latents)
    
    # 다양한 timestep에서 학습
    timesteps = torch.randint(0, model.scheduler.config.num_train_timesteps, (batch_size,), device=device)
    noisy_latents = model.scheduler.add_noise(target_latents, noise, timesteps)
    
    # 5. 텍스트 조건화를 위한 임베딩 생성
    text_inputs = model.tokenizer(
        truncated_texts,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt"
    ).to(device)
    
    with torch.no_grad():
        text_embeddings = model.text_encoder(**text_inputs)[0]
    
    # 데이터 타입 일치시키기
    if model.unet.dtype == torch.float16:
        noisy_latents = noisy_latents.half()
        text_embeddings = text_embeddings.half()
    else:
        noisy_latents = noisy_latents.float()
        text_embeddings = text_embeddings.float()
    
    # 6. UNet으로 노이즈 예측 (LoRA 적용된 상태)
    model_pred = model.unet(
        noisy_latents,
        timesteps,
        encoder_hidden_states=text_embeddings
    ).sample
    
    # 7. 기본 diffusion 손실
    diffusion_loss = F.mse_loss(model_pred, noise)
    
    # 8. 평가 기반 가중치 적용
    satisfaction_penalty = (10.0 - overall_scores) / 10.0
    text_match_penalty = (10.0 - text_matches) / 10.0
    
    # 음질이 나쁜 경우 추가 페널티
    quality_penalty = torch.tensor([
        2.0 if quality == "Bad" else 1.0 
        for quality in batch["audio_quality"]
    ], device=device)
    
    # 개선사항이 있는 경우 가중치 증가
    improvement_weight = torch.tensor([
        1.5 if improvement.strip() else 1.0 
        for improvement in improvements
    ], device=device)
    
    # 9. 감정 기반 가중치 (NEW!)
    emotion_penalty = torch.tensor([
        2.0 - emotion_score  # 감정 신뢰도가 낮으면 더 학습
        for emotion_score in emotion_scores.cpu()
    ], device=device)
    
    # 10. 종합 가중치 계산 (감정 정보 포함)
    total_weights = (
        satisfaction_penalty * 1.0 +          # 만족도가 낮으면 더 학습
        text_match_penalty * 0.8 +            # 텍스트 매치가 낮으면 더 학습  
        quality_penalty * 0.5 +               # 음질이 나쁘면 더 학습
        improvement_weight * 0.3 +            # 개선사항이 있으면 더 학습
        emotion_penalty * 0.4                 # 감정 신뢰도가 낮으면 더 학습 (NEW!)
    ) * emotion_weights  # 감정별 가중치 적용
    
    # 11. 최종 손실 (배치 평균)
    weighted_loss = diffusion_loss * torch.mean(total_weights)
    
    return weighted_loss

def train_lora_with_emotion(model, train_loader, num_epochs, learning_rate, device, save_dir):
    """
    감정 정보를 활용한 LoRA 모델 학습
    """
    # 옵티마이저 설정
    trainable_params = [p for p in model.unet.parameters() if p.requires_grad]
    print(f"Trainable parameters: {len(trainable_params)}")
    
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=1e-4)
    
    # 스케줄러 설정
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps
    )
    
    model.unet.train()
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, batch in enumerate(progress_bar):
            optimizer.zero_grad()
            
            # 감정 정보를 포함한 손실 계산
            loss = improved_lora_loss_with_emotion(model, batch, device)
            
            # 그래디언트 체크
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Invalid loss at batch {batch_idx}: {loss.item()}")
                continue
                
            loss.backward()
            
            # 그래디언트 클리핑
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
            
            # 진행률 표시 (감정 정보 포함)
            emotions_in_batch = batch["dominant_emotion"][:3]  # 처음 3개만 표시
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}', 
                'lr': f'{scheduler.get_last_lr()[0]:.2e}',
                'emotions': '/'.join(emotions_in_batch)
            })
        
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1} completed - Average Loss: {avg_loss:.4f}")
        
        # 모델 저장 (매 에포크마다)
        epoch_save_dir = os.path.join(save_dir, f"epoch_{epoch+1}")
        ensure_dir(epoch_save_dir)
        model.unet.save_pretrained(epoch_save_dir)
        print(f"Model saved to: {epoch_save_dir}")

def main():
    parser = argparse.ArgumentParser(description="AudioLDM LoRA Training with Emotion")
    parser.add_argument("--train_json", default="data/z_rating_data_train.json", help="Training data JSON path")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--save_dir", default="weight/lora_emotion_weights", help="Directory to save LoRA weights")
    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # AudioLDM 모델 로드
    print("Loading AudioLDM model...")
    model = AudioLDMPipeline.from_pretrained(
        "cvssp/audioldm",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    
    # LoRA 설정
    lora_params = {
        "r": 16,                # LoRA rank
        "lora_alpha": 32,       # LoRA alpha
        "lora_dropout": 0.1     # LoRA dropout
    }
    
    model = setup_lora_model(model, lora_params)
    print("LoRA setup completed!")
    
    # 데이터셋 및 데이터로더 설정
    print(f"Loading emotion-enhanced dataset from: {args.train_json}")
    dataset = AudioRatingEmotionDataset(args.train_json)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    
    print(f"Dataset loaded: {len(dataset)} samples")
    
    # 감정 분포 확인
    emotions = [item["dominant_emotion"] for item in dataset.data]
    emotion_counts = {}
    for emotion in emotions:
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
    print(f"Emotion distribution: {emotion_counts}")
    
    # 학습 시작
    ensure_dir(args.save_dir)
    print("Starting emotion-enhanced LoRA training...")
    
    train_lora_with_emotion(
        model=model,
        train_loader=dataloader,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        device=device,
        save_dir=args.save_dir
    )
    
    print("Training completed!")
    print(f"Final LoRA weights saved to: {args.save_dir}")

if __name__ == "__main__":
    main()