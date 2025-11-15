슬라이딩 윈도우로 전처리 후 205개만 데이터 분석용으로 사용
- 200번째까지 데이터 training, lora 용도
- 201~205번째 오디오를 발표 test 용도

baseline: 감정 X / LLM api X / AudioLDM1, LoRA X 
lora: 감정 X / LLM api X / AudioLDM1, LoRA O

- 201~205번째 오디오를 baseline, Model1, Model2 비교

# ============================================== #

파일명

전체 데이터 /data/a_data.txt

a_sliding_window.py: 데이터 감정 태깅한 json 생성
아래 데이터에서 감정을 태깅할 필요 없음.
생성한 0~199번 데이터 /data/a_data_train.json
생성한 200~232번 데이터 /data/a_data_test.json

baseline_generate.py
baseline 실행 후 0~199번 데이터 output/baseline/train/
=> 이때 data/a_data_train 데이터의 n번째 데이터를 넣고 n번째 오디오를 1개씩 생성
baseline 실행 후 200~232번 데이터 output/baseline/test/
=> 이때 data/a_data_test 데이터의 n번째 데이터를 넣고 n번째 오디오를 1개씩 생성

a_data_train.json을 rating_data_train.json에 데이터 복붙 후,
rating_data_train.json에 오디오 경로, 인간 rating 을 추가(직접 오디오를 듣고 주관적 평가 점수 추가)
rating_data_train.json을 넣어서 lora_training.py로 weight/ 생성
lora_generate.py 실행 후 200~232번 데이터를 넣어 output/lora/test/ 에 오디오 생성

# Train 데이터 (0~199번)
!python baseline_generate.py --single_index 0    # 첫 번째
!python baseline_generate.py --single_index 1    # 두 번째  
!python baseline_generate.py --single_index 50   # 50번째

# Test 데이터 (200~232번) 
!python baseline_generate.py --single_index 200  # 첫 번째 테스트
!python baseline_generate.py --single_index 201  # 두 번째 테스트
!python baseline_generate.py --single_index 205  # 발표용 (README 기준)

# Test 데이터만 지원 (200~232번) - LoRA weights 경로 필수!
!python lora_generate.py --single_index 200 --lora_weights weight/lora_weights
!python lora_generate.py --single_index 201 --lora_weights weight/lora_weights  
!python lora_generate.py --single_index 205 --lora_weights weight/lora_weights

# rating_data_train.json 형태
{
  "scene_id": 0,
  "text": "어두운 숲속을 걸었다",
  "audio_path": "output/baseline/train/000_baseline.wav",
  "overall_score": 7.5,        // 전체 만족도 (1~10)
  "text_match": 6.0,          // 텍스트 적합성 (1~10)  
  "audio_quality": "Good",     // 음질 (Good/Bad)
  "improvement": "더 어두운 분위기 필요"  // 선택적 피드백
}

# LoRA가 오디오를 해석하는 방식
1. LoRA는 오디오 파일을 직접 "읽지" 않고
2. AudioLDM의 VAE가 오디오를 잠재 표현으로 변환
3. LoRA는 잠재 공간에서 텍스트→오디오 매핑을 학습
4. 인간 평가 점수로 학습 방향을 조정

rating_data_train_short.json을 만들어서 scene 0~24까지(25개)만 고려하게 하고
LoRA 우선 실행해서 weights 생성
baseline의 scene 200~205번 생성, LoRA를 이용한 scene 200~205번 생성하여 두 방식 비교
!python lora_training.py \
  --train_json data/rating_data_train_short.json \
  --num_epochs 5 \
  --batch_size 2 \
  --learning_rate 1e-4

# 추가 성능 개선 1: 감정 태깅했던 거 다시 넣어서 만들기
1) z_sliding_window_emotion.py로 z_rating_data_train.json을 만들고 scene_id 0~199만 남김.
2) z_rating_data_train.json을 수정해서 rating_data_train의 scene_id가 맞는 것끼리 rating_data_train에 있는 다른 컬럼을 붙임.
3) lora_training.py의 로직에서 emotion 관련을 추가로 처리하는 lora_emotion_training.py를 생성.
!python lora_emotion_training.py \
  --train_json data/z_rating_data_train_short.json \
  --num_epochs 5 \
  --batch_size 2 \
  --learning_rate 1e-4
4) weight/lora_emotion_weights/에 weight 생성.
이후 동일하게 lora_generate.py 로직과 완전히 동일한 lora_emotion_generate.py를 만들고, 오디오의 저장 위치만 output/lora_emotion/test/에 저장되게 함.
