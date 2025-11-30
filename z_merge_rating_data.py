# z_merge_rating_data.py
# z_data_emotion_train.json과 rating_data_train_short.json을 scene_id 기준으로 병합

import json
import os

def load_json(path):
    """JSON 파일 로드"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(data, path):
    """JSON 파일 저장"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def merge_rating_data():
    """
    z_data_emotion_train.json (감정 태깅 데이터)와 
    rating_data_train_short.json (평가 데이터)를 scene_id 기준으로 병합
    """
    
    # 입력 파일들
    emotion_file = "data/z_data_emotion_train.json"
    rating_file = "data/rating_data_train_short.json"
    
    # 출력 파일
    output_file = "data/z_rating_data_train.json"
    
    print("🔄 Loading data files...")
    
    # 데이터 로드
    emotion_data = load_json(emotion_file)
    rating_data = load_json(rating_file)
    
    # scene_id 0~199만 필터링 (training 데이터)
    emotion_filtered = [item for item in emotion_data if item["scene_id"] <= 199]
    
    print(f"📊 Emotion data: {len(emotion_filtered)} items (scene_id 0~199)")
    print(f"📊 Rating data: {len(rating_data)} items")
    
    # rating_data를 scene_id 기준으로 딕셔너리로 변환 (빠른 검색)
    rating_dict = {item["scene_id"]: item for item in rating_data}
    
    # 병합된 데이터 생성
    merged_data = []
    
    for emotion_item in emotion_filtered:
        scene_id = emotion_item["scene_id"]
        
        # 새로운 항목 생성 (emotion 데이터 기반)
        merged_item = {
            "scene_id": scene_id,
            "text": emotion_item["text"],
            "dominant_emotion": emotion_item["dominant_emotion"],
            "score": emotion_item["score"]
        }
        
        # 해당 scene_id가 rating_data에 있으면 추가 정보 병합
        if scene_id in rating_dict:
            rating_item = rating_dict[scene_id]
            merged_item.update({
                "audio_path": rating_item["audio_path"],
                "overall_score": rating_item["overall_score"],
                "text_match": rating_item["text_match"],
                "audio_quality": rating_item["audio_quality"],
                "improvement": rating_item["improvement"]
            })
            print(f"✅ Merged scene_id {scene_id}: emotion + rating data")
        else:
            # rating 데이터가 없으면 기본값 설정
            merged_item.update({
                "audio_path": f"output/baseline/train/{scene_id:03d}_baseline.wav",
                "overall_score": 0,
                "text_match": 0,
                "audio_quality": "Bad",
                "improvement": "평가 필요"
            })
            print(f"⚠️ Scene_id {scene_id}: only emotion data (added default rating)")
        
        merged_data.append(merged_item)
    
    # 결과 저장
    save_json(merged_data, output_file)
    
    print(f"\n🎉 Merged data saved to: {output_file}")
    print(f"📊 Total items: {len(merged_data)}")
    
    # 병합 통계 출력
    with_rating = sum(1 for item in merged_data if item["overall_score"] > 0)
    without_rating = len(merged_data) - with_rating
    
    print(f"📈 Items with rating data: {with_rating}")
    print(f"📉 Items with default rating: {without_rating}")
    
    # 샘플 출력
    print(f"\n📝 Sample merged data:")
    for i in range(min(3, len(merged_data))):
        item = merged_data[i]
        print(f"  Scene {item['scene_id']}: {item['dominant_emotion']} (score: {item['score']:.3f})")
        print(f"    Audio: {item['audio_path']}")
        print(f"    Rating: {item['overall_score']} | Quality: {item['audio_quality']}")

if __name__ == "__main__":
    merge_rating_data()