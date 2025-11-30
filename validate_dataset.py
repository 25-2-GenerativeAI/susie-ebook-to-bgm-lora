#!/usr/bin/env python3
# validate_dataset.py
# 🔥 Rating Data Train 검증 스크립트 - 모든 invalid 샘플 찾기

import json
import os
import librosa
import numpy as np
import soundfile as sf

def validate_dataset(json_path="data/rating_data_train.json"):
    """데이터셋 완전 검증"""
    print(f"🔥 Validating dataset: {json_path}")
    
    if not os.path.exists(json_path):
        print(f"❌ JSON file not found: {json_path}")
        return
    
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ Failed to load JSON: {e}")
        return
    
    print(f"📊 Total samples in dataset: {len(data)}")
    print("-" * 60)
    
    valid = 0
    invalid = 0
    issues = {
        "missing_file": 0,
        "too_short": 0, 
        "silent": 0,
        "empty_text": 0,
        "load_error": 0,
        "type_error": 0
    }
    
    for i, item in enumerate(data):
        sample_valid = True
        sample_issues = []
        
        # 기본 필드 체크
        audio_path = item.get("audio_path", "")
        text = item.get("text", "")
        overall_score = item.get("overall_score", 0)
        text_match = item.get("text_match", 0)
        audio_quality = item.get("audio_quality", "Bad")
        
        # Text 검증
        if not text or len(text.strip()) == 0:
            sample_issues.append("empty_text")
            issues["empty_text"] += 1
            sample_valid = False
        
        # Score type 검증  
        try:
            float(overall_score)
            float(text_match)
        except (ValueError, TypeError):
            sample_issues.append(f"type_error: overall_score={type(overall_score)}, text_match={type(text_match)}")
            issues["type_error"] += 1
            sample_valid = False
        
        # Audio path 검증
        if not audio_path:
            sample_issues.append("missing_audio_path")
            sample_valid = False
        elif not os.path.exists(audio_path):
            sample_issues.append(f"missing_file: {audio_path}")
            issues["missing_file"] += 1
            sample_valid = False
        else:
            # Audio 로드 및 검증
            try:
                audio, sr = librosa.load(audio_path, sr=16000)
                
                if len(audio) < 1000:
                    sample_issues.append(f"too_short: {len(audio)} samples ({len(audio)/16000:.3f}s)")
                    issues["too_short"] += 1
                    sample_valid = False
                
                if np.all(audio == 0):
                    sample_issues.append("silent_audio")
                    issues["silent"] += 1
                    sample_valid = False
                
                if np.any(np.isnan(audio)) or np.any(np.isinf(audio)):
                    sample_issues.append("invalid_audio_values")
                    sample_valid = False
                    
            except Exception as e:
                sample_issues.append(f"load_error: {str(e)[:50]}")
                issues["load_error"] += 1
                sample_valid = False
        
        if sample_valid:
            valid += 1
            if i < 5:  # 처음 5개만 출력
                print(f"✅ [{i:3d}] VALID: {text[:50]}...")
        else:
            invalid += 1
            print(f"❌ [{i:3d}] INVALID: {', '.join(sample_issues)}")
            if text:
                print(f"     Text: {text[:80]}...")
            if audio_path:
                print(f"     Path: {audio_path}")
            print()
    
    print("=" * 60)
    print(f"📊 SUMMARY")
    print(f"✅ VALID samples: {valid}")
    print(f"❌ INVALID samples: {invalid}")
    print(f"📈 Success rate: {valid/(valid+invalid)*100:.1f}%")
    print()
    
    print("🔍 ISSUE BREAKDOWN:")
    for issue_type, count in issues.items():
        if count > 0:
            print(f"  {issue_type}: {count}")
    
    if valid == 0:
        print("\n🚨 CRITICAL: NO VALID SAMPLES FOUND!")
        print("🔧 This explains why training fails with 'ALL BATCHES FAILED'")
    elif valid < 5:
        print(f"\n⚠️  WARNING: Only {valid} valid samples - training will be unstable")
    
    return valid, invalid, issues

if __name__ == "__main__":
    # 현재 디렉토리 확인
    print(f"📁 Current directory: {os.getcwd()}")
    print(f"📁 Files in current directory:")
    for item in sorted(os.listdir(".")):
        if os.path.isfile(item):
            print(f"   📄 {item}")
        else:
            print(f"   📁 {item}/")
    print()
    
    # 데이터셋 검증 실행
    validate_dataset()