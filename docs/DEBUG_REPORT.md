# 🛠️ 프로젝트 정상화 디버깅 리포트
**작성일:** 2024-XX-XX
**목표:** 데이터 파이프라인의 각 단계별 무결성 검증 및 모델 학습 실패 원인 규명

## 1단계: 데이터 생성 (Data Generation)
* **테스트 스크립트:** `tests/verify_01_generation.py`
* **검증 항목:**
    * [ ] Glucose/Insulin 값이 생리학적 범위(80~200, 10~100) 내에 있는가?
    * [ ] Parameter(si, sigma)가 음수나 0이 아닌가?
    * [ ] 데이터에 `NaN`이나 `Inf`가 없는가?
* **테스트 결과:**
    * (여기에 실행 로그 요약 또는 'PASS'/'FAIL' 작성)
* **조치 사항:**
    * (문제가 발견되면 수정한 내용 기록)

## 2단계: 데이터 로드 및 전처리 (DataLoader & Preprocessing)
* **테스트 스크립트:** `tests/verify_02_loader.py`
* **검증 항목:**
    * [ ] DataLoader가 에러 없이 배치를 생성하는가?
    * [ ] 배치 데이터($X, Y$)의 스케일이 0~1 사이인가, 아니면 Raw Scale(100단위)인가?
    * [ ] 정답($P$) 데이터가 모델이 예측 가능한 범위(0~2)인가?
    * [ ] Normalizer가 의도한 대로(Identity vs Scaling) 동작하는가?
* **테스트 결과:**
    * ...

## 3단계: 모델 초기화 및 출력 (Model Initialization)
* **테스트 스크립트:** `tests/verify_03_model_init.py`
* **검증 항목:**
    * [ ] 모델이 에러 없이 생성되는가?
    * [ ] 초기 입력에 대해 `0.000`이 아닌 유효한 값을 출력하는가?
    * [ ] 마지막 레이어 Bias 초기화(0.5)가 적용되었는가?
* **테스트 결과:**
    * ...

## 4단계: 학습 단계 (Training Step)
* **테스트 스크립트:** `tests/verify_04_train_step.py`
* **검증 항목:**
    * [ ] Forward Pass에서 Loss가 계산되는가?
    * [ ] Backward Pass 후 Gradient가 0이 아닌 유효한 값을 가지는가?
    * [ ] Optimizer Step 후 가중치가 실제로 변하는가?
* **테스트 결과:**
    * ...