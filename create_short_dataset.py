#!/usr/bin/env python3
"""
Create rating_data_train_short.json with first 25 samples for quick LoRA testing
"""

import json

def create_short_dataset():
    """Extract first 25 samples from rating_data_train.json"""
    
    # Read full training data
    with open('data/rating_data_train.json', 'r', encoding='utf-8') as f:
        full_data = json.load(f)
    
    # Extract first 25 samples (scene_id 0-24)
    short_data = full_data[:25]
    
    # Save short dataset
    with open('data/rating_data_train_short.json', 'w', encoding='utf-8') as f:
        json.dump(short_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Short dataset created: {len(short_data)} samples")
    print(f"📁 Scene IDs: {short_data[0]['scene_id']} ~ {short_data[-1]['scene_id']}")
    print("📄 Saved to: data/rating_data_train_short.json")
    print("\n🎯 Next steps:")
    print("1. Generate baseline audio for scenes 0-24")
    print("2. Add manual ratings to rating_data_train_short.json")
    print("3. Run lora_training.py with short dataset")

if __name__ == "__main__":
    create_short_dataset()