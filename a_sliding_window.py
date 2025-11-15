import os
import re
import json
import numpy as np
import torch
from nltk.tokenize import sent_tokenize
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# 환경 설정
# -------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    print("⚠️ WARNING: No GPU detected.")

# 감정 분류 모델 로드
try:
    emotion_classifier = pipeline(
        "text-classification",
        model="SamLowe/roberta-base-go_emotions",
        return_all_scores=True,
        device=0 if device == "cuda" else -1
    )
    print("✅ Emotion Classifier loaded.")
except Exception as e:
    emotion_classifier = None
    print(f"🚨 Emotion Classifier failed: {e}")

# -------------------------------
# 파라미터
# -------------------------------
WINDOW_SIZE = 5
K_STD_DEV = 1.5
EMOTION_THRESHOLD = 0.5

# -------------------------------
# 함수 정의
# -------------------------------
def load_text(file_path: str) -> str:
    """텍스트 파일 불러오기"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found.")
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def get_emotion_vector(text):
    """텍스트 감정 벡터 추출"""
    emotions = emotion_classifier(text, top_k=None, truncation=True)
    return np.array(sorted([e['score'] for e in emotions]))

def segment_text_by_emotion(text, window_size, k_std):
    """감정 변화 기반으로 씬 분할"""
    text = re.sub(r'\s+', ' ', text).strip()
    sentences = sent_tokenize(text)

    if len(sentences) < window_size * 2:
        return [text]

    # 슬라이딩 윈도우 기반 감정 벡터 계산
    emotion_vectors = [
        get_emotion_vector(" ".join(sentences[i:i+window_size]))
        for i in range(len(sentences) - window_size + 1)
    ]

    # 감정 변화량 계산
    change_scores = np.array([
        1 - cosine_similarity([v1], [v2])[0][0]
        for v1, v2 in zip(emotion_vectors, emotion_vectors[1:])
    ])

    threshold = np.mean(change_scores) + k_std * np.std(change_scores)
    split_indices = [i for i, score in enumerate(change_scores) if score > threshold]

    if not split_indices:
        return [text]

    # 컷팅
    scene_chunks, start_idx = [], 0
    for idx in split_indices:
        split_point = idx + int(window_size / 2)
        if split_point > start_idx:
            scene_chunks.append(" ".join(sentences[start_idx:split_point]))
            start_idx = split_point
    scene_chunks.append(" ".join(sentences[start_idx:]))

    # 10 단어 이상만 유지
    final_chunks = [chunk for chunk in scene_chunks if len(chunk.split()) > 10]
    return final_chunks

def analyze_emotional_intensity(text_chunk):
    """씬 단위 감정 분석"""
    emotions = emotion_classifier(text_chunk, top_k=None, truncation=True)
    dominant = max(emotions, key=lambda x: x['score'])
    return dominant['label'], dominant['score']

# -------------------------------
# 실행부
# -------------------------------
def main(input_file="data/a_data.txt", output_path="data/a_data_train.json"):
    if not emotion_classifier:
        print("🚨 Emotion classifier not available.")
        return

    # 텍스트 파일 불러오기
    text_to_process = load_text(input_file)

    # 씬 분할
    emotional_chunks = segment_text_by_emotion(text_to_process, WINDOW_SIZE, K_STD_DEV)

    results = []
    for i, chunk in enumerate(emotional_chunks):
        dominant_emotion, score = analyze_emotional_intensity(chunk)
        results.append({
            "scene_id": i,
            "text": chunk
        })

    # JSON 저장
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"✅ Results saved to {output_path}")

# -------------------------------
# 실행
# -------------------------------
if __name__ == "__main__":
    import nltk

    # 필요한 리소스 다운로드
    nltk.download("punkt")
    nltk.download("punkt_tab")

    main()
