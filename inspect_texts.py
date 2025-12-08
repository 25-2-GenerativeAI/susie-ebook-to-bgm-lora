#!/usr/bin/env python3
"""
데이터셋의 텍스트 필드 검사 스크립트
문제가 있는 텍스트를 찾아내서 리포팅함
"""

import json
import sys

def inspect_texts(json_path):
    """JSON 파일의 모든 text 필드를 검사"""
    
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    print(f"🔍 Inspecting {json_path}")
    print(f"Total samples: {len(data)}")
    print("=" * 70)
    
    issues = {
        "none": [],
        "empty": [],
        "non_string": [],
        "korean": [],
        "too_long": [],
    }
    
    for idx, item in enumerate(data):
        text = item.get("text", None)
        
        # 1. None 체크
        if text is None:
            issues["none"].append(idx)
            continue
        
        # 2. 타입 체크
        if not isinstance(text, str):
            issues["non_string"].append((idx, type(text).__name__))
            continue
        
        # 3. 빈 문자열 체크
        if len(text.strip()) == 0:
            issues["empty"].append(idx)
            continue
        
        # 4. 한글 체크 (실제 한글만, 아포스트로피는 무시)
        has_korean = any(0xAC00 <= ord(c) < 0xD7A3 for c in text)
        if has_korean:
            issues["korean"].append((idx, text[:50]))
            continue
        
        # 5. 길이 체크
        if len(text) > 500:
            issues["too_long"].append((idx, len(text)))
    
    # 결과 출력
    print("\n📊 Issues Found:")
    print("-" * 70)
    
    if issues["none"]:
        print(f"❌ None values: {len(issues['none'])} samples")
        print(f"   Indices: {issues['none'][:10]}")
    
    if issues["empty"]:
        print(f"❌ Empty strings: {len(issues['empty'])} samples")
        print(f"   Indices: {issues['empty'][:10]}")
    
    if issues["non_string"]:
        print(f"❌ Non-string types: {len(issues['non_string'])} samples")
        for idx, dtype in issues["non_string"][:5]:
            print(f"   [{idx}] type={dtype}")
    
    if issues["korean"]:
        print(f"❌ Non-English text: {len(issues['korean'])} samples")
        for idx, text in issues["korean"][:5]:
            print(f"   [{idx}] {text}")
    
    if issues["too_long"]:
        print(f"⚠️  Long texts (>500 chars): {len(issues['too_long'])} samples")
        for idx, length in issues["too_long"][:5]:
            print(f"   [{idx}] length={length}")
    
    total_issues = sum(len(v) if isinstance(v, list) else len(v) 
                      for v in issues.values())
    
    print("-" * 70)
    if total_issues == 0:
        print("✅ All texts are valid!")
    else:
        print(f"⚠️  Total issues: {total_issues}")
    
    print("\n📋 Sample texts:")
    print("-" * 70)
    for idx, item in enumerate(data[:5]):
        text = item.get("text", "None")
        print(f"[{idx}] {text[:60]}")


if __name__ == "__main__":
    json_path = "data/rating_data_train.json"
    if len(sys.argv) > 1:
        json_path = sys.argv[1]
    
    inspect_texts(json_path)
