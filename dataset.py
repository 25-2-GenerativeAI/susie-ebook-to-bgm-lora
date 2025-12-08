"""
AudioLDM1 LoRA 훈련용 데이터셋
JSON 포맷: [{"audio_path": "...", "text": "...", "overall_score": 8.0, "text_match": 7.5}, ...]
"""

import torch
import librosa
import numpy as np
import json


class AudioRatingDataset(torch.utils.data.Dataset):
    """
    오디오 + 텍스트 + 평가 점수 데이터셋
    """
    
    def __init__(self, json_path, sr=16000, max_length=96000):
        """
        Args:
            json_path: 훈련 데이터 JSON 파일
            sr: 샘플링 레이트
            max_length: 최대 오디오 길이 (샘플)
        """
        self.sr = sr
        self.max_length = max_length
        
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        
        print(f"[Dataset] Loaded {len(self.data)} samples from {json_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # ✅ CRITICAL: 텍스트는 반드시 string이어야 함
        # (CLAP 인코더가 string list만 받음)
        text = item.get("text", "")
        
        # None 체크
        if text is None:
            text = ""
        
        # String으로 변환 (list, int, float 등 다른 타입이 들어올 수 있음)
        text = str(text).strip()
        
        # 최대 500자 제한
        text = text[:500]
        
        # 빈 문자열이면 기본값 설정
        if len(text) == 0:
            text = "ambient music"
        
        # 한국어 텍스트는 CLAP이 처리 못함 → 영어로 변환 필요
        # (지금은 간단히 fallback, 실제로는 번역 모듈 필요)
        # NOTE: 아포스트로피(')와 특수 문자는 OK, 실제 한글/중국어만 필터링
        try:
            # 실제 한글 감지: U+AC00 이상
            for char in text:
                if ord(char) > 0xAC00 and ord(char) < 0xD7A3:  # 한글 범위
                    raise UnicodeDecodeError('ascii', b'', 0, 1, 'korean detected')
        except UnicodeDecodeError:
            # 한국어면 기본값으로 대체
            print(f"[Dataset] Korean text detected, using default: {text[:30]}")
            text = "ambient soundscape"

        # 오디오 로드 및 패딩/자르기
        audio_path = item.get("audio_path", "")
        try:
            audio, _ = librosa.load(audio_path, sr=self.sr, duration=6.0)
        except:
            print(f"[Dataset] Failed to load {audio_path}, using silence")
            audio = np.zeros(self.sr * 6)
        
        if len(audio) < self.max_length:
            audio = np.pad(audio, (0, self.max_length - len(audio)))
        else:
            audio = audio[:self.max_length]

        # 평가 점수 (반드시 float로 변환)
        overall_score = float(item.get("overall_score", 5.0))
        text_match = float(item.get("text_match", 5.0))

        return {
            "text": text,  # ✅ 반드시 string
            "audio": torch.FloatTensor(audio),
            "overall_score": overall_score,
            "text_match": text_match,
        }
