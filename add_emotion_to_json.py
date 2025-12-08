#!/usr/bin/env python3
"""
기존 rating_data_train.json에 감정(emotion) 정보를 추가하는 스크립트
=========================================================================

기능:
- 기존 JSON 파일의 각 씬(scene)에 대해 감정 분석 수행
- dominant_emotion과 emotion_score 필드 추가
- 원본 JSON 구조 유지하면서 감정 정보만 추가
"""

import json
import torch
import numpy as np
from transformers import pipeline
from tqdm import tqdm

# -------------------------------
# 환경 설정
# -------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

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
    exit(1)


def get_emotion_vector(text):
    """
    텍스트의 전체 감정 분포 벡터 반환 (28차원)
    
    Args:
        text: 분석할 텍스트
    
    Returns:
        np.ndarray: 28차원 감정 확률 벡터
    """
    emotions = emotion_classifier(text, top_k=None, truncation=True)
    # 점수 순으로 정렬하여 일관된 순서 유지
    emotion_vector = np.array(sorted([e['score'] for e in emotions]))
    return emotion_vector


def analyze_emotion(text):
    """
    텍스트의 감정을 분석하여 가장 강한 감정과 점수 반환
    
    Args:
        text: 분석할 텍스트
    
    Returns:
        tuple: (dominant_emotion, emotion_score)
    """
    emotions = emotion_classifier(text, top_k=None, truncation=True)
    dominant = max(emotions, key=lambda x: x['score'])
    return dominant['label'], dominant['score']


def add_emotions_to_json(input_path, output_path):
    """
    JSON 파일에 감정 정보 추가
    - dominant_emotion: 가장 강한 감정 레이블
    - emotion_score: 가장 강한 감정의 점수
    - emotion_vector: 28차원 전체 감정 분포 벡터
    
    Args:
        input_path: 입력 JSON 파일 경로
        output_path: 출력 JSON 파일 경로
    """
    print(f"Loading JSON from {input_path}...")
    
    # JSON 파일 로드
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} scenes")
    
    # 각 씬에 대해 감정 분석 수행
    print("Analyzing emotions...")
    for item in tqdm(data):
        text = item.get("text", "")
        
        # 전체 감정 벡터 추출 (28차원)
        emotion_vector = get_emotion_vector(text)
        
        # 가장 강한 감정 추출
        dominant_emotion, emotion_score = analyze_emotion(text)
        
        # 감정 정보 추가
        item["emotion_vector"] = emotion_vector.tolist()  # numpy array를 list로 변환
        item["dominant_emotion"] = dominant_emotion
        item["emotion_score"] = float(emotion_score)
    
    # 결과 저장
    print(f"Saving results to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Complete! Emotion data added to {output_path}")
    
    # 감정 분포 출력
    print("\n📊 Emotion Distribution:")
    emotion_counts = {}
    for item in data:
        emotion = item["dominant_emotion"]
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
    
    for emotion, count in sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(data)) * 100
        print(f"  {emotion:15s}: {count:4d} ({percentage:5.1f}%)")
    
    # 벡터 차원 확인
    print(f"\n✅ Emotion vector dimension: {len(data[0]['emotion_vector'])}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Add emotion analysis to existing JSON dataset")
    parser.add_argument(
        "--input",
        type=str,
        default="data/rating_data_train.json",
        help="Input JSON file path"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/rating_data_train_with_emotion.json",
        help="Output JSON file path"
    )
    
    args = parser.parse_args()
    
    add_emotions_to_json(args.input, args.output)


if __name__ == "__main__":
    main()
