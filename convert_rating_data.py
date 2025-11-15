#!/usr/bin/env python3
"""
Convert a_data_train.json to rating_data_train.json with 3+1 evaluation structure
"""

import json

def convert_to_rating_format():
    """Convert training data to rating format with empty evaluation fields"""
    
    # Read original training data
    with open('data/a_data_train.json', 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    
    # Convert to rating format
    rating_data = []
    for item in train_data:
        rating_item = {
            "scene_id": item["scene_id"],
            "text": item["text"],
            "audio_path": f"output/baseline/train/{item['scene_id']:03d}_baseline.wav",
            "overall_score": None,  # 1-10 scale (empty for manual rating)
            "text_match": None,     # 1-10 scale (empty for manual rating)
            "audio_quality": None,  # "Good" or "Bad" (empty for manual rating)
            "improvement": None     # Text feedback (empty for manual rating)
        }
        rating_data.append(rating_item)
    
    # Save converted data
    with open('data/rating_data_train.json', 'w', encoding='utf-8') as f:
        json.dump(rating_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Converted {len(rating_data)} training samples to rating format")
    print("📁 Saved to: data/rating_data_train.json")
    print("\n🎯 3+1 Evaluation Structure:")
    print("  - overall_score: 1-10 (overall BGM quality)")
    print("  - text_match: 1-10 (how well BGM matches text)")
    print("  - audio_quality: Good/Bad (technical quality)")
    print("  - improvement: Text feedback for LoRA training")
    print(f"\n📊 Ready for manual rating of {len(rating_data)} baseline audio files")

if __name__ == "__main__":
    convert_to_rating_format()