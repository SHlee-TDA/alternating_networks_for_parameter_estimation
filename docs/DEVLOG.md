# 개발 로그 (Development Log)

## 2025-11-18: NIH 데이터 로더 구현 및 수치해석 이슈 해결 (Final)

### 1. 주요 변경 사항 (Phase 2 완료)
* **RealOGTTDataLoader 구현:** NIH 데이터 로드, 15분 시점 제외, 결측치(NaN) 포함 샘플 삭제 정책 적용.
* **noise_calibration.py 구현:** 실제 데이터 기반의 $\sigma_{emp}$ (경험적 노이즈 표준편차) 추정 파이프라인 구축.
* **systems/base_system.py 확장:** SDE 구현을 위한 `drift_func`와 `diffusion_func` 인터페이스 추가.

### 2. 트러블슈팅 (Troubleshooting) - Critical Fix
* **이슈:** `scipy.integrate.solve_ivp` 실행 시 `The truth value of an array with more than one element is ambiguous` 에러 발생.
* **원인:**
    1.  `calculate_ogtt_flux`에서 시간 $t$가 벡터로 들어올 때 Python 기본 `if`문 사용 (NumPy 호환성 문제).
    2.  `systems/ogtt_simul.py`의 `simulate` 함수에서 `t_eval` (NumPy Array)을 `== None`으로 비교함 (객체 정체성 비교 오류).
* **해결:**
    1.  `calculate_ogtt_flux`: `if/elif` 구조를 `np.select` 기반의 벡터화 연산으로 변경하여 해결.
    2.  `simulate`: `if t_eval == None`을 `if t_eval is None`으로 수정하여 해결.
* **교훈:** NumPy/SciPy 환경에서 조건문과 객체 비교(`==` vs `is`) 사용 시 항상 입력 차원(스칼라/벡터)과 객체 유형(Array/None)을 확인할 것.



## 2025-11-19: SDE Solver 구현 및 수치 안정화 (Phase 3)

### 1. 변경 사항
* **SDE Solver Core 구현:**
  * `utils.py`: Euler-Maruyama Solver 구현 (1분 간격 고해상도 시뮬레이션).
  * `systems/ogtt_simul.py`: `diffusion_func` 구현 (4x4 대각 확산 행렬, $\sigma(t)$ 선형 보간 적용).
  * `noise_calibration.py`: 각 시점별(t=0, 30...) 잔차 분포 분석 기능 추가 및 상한값($Y_{max}$) 계산 로직 추가.

### 2. 트러블슈팅 (Troubleshooting) - Numerical Instability
* **이슈:** SDE Solver 테스트 중 `RuntimeWarning: overflow encountered` 및 결과값 `NaN` 발생.
* **원인:**
  * SDE의 확률적 노이즈가 상태 변수(특히 Glucose $G$)를 물리적 범위를 벗어난 값(음수 또는 극단적 양수)으로 밀어냄.
  * `calculate_GF` 함수의 $(G - shGF)^{16}$ 항이 오버플로우를 일으키며 `inf` 및 `nan`을 유발.
* **해결:** **물리적 제약 조건(Physical Clamping)** 도입.
  * **Lower Bound:** $10^{-6}$ (음수 방지 및 0으로 나누기 방지).
  * **Upper Bound:** 실제 데이터 관측 최댓값의 110% ($1.1 \times Max_{obs}$).
  * `noise_calibration.py`에서 상한값을 계산하여 `calibrated_sigmas.json`에 저장하고, `euler_maruyama`에서 매 스텝마다 이를 적용.
* **교훈:** 생물학적/물리적 모델의 SDE 시뮬레이션 시, 상태 변수가 유효 범위 내에 머물도록 강제하는 안전장치(Clamping)가 필수적임.


## 2025-11-19: 데이터 증강 파이프라인 통합 (Phase 4)

### 1. 변경 사항
* **Data-Driven Sampling:**
  * `distribution_analysis.py`: 실제 NIH 데이터의 $G_0, I_0, S_I, \sigma$ 분포를 Log-Normal로 피팅하고 `distribution_params.json` 생성.
  * `data_loader.py`: `sample_from_lognorm` 함수 추가 (Rejection Sampling으로 양수 보장).
* **DataGenerator 업그레이드:**
  * `USE_SDE` 플래그에 따라 Euler-Maruyama Solver 호출.
  * `USE_LAGRANGIAN` 플래그에 따라 Drift Term(=도함수) Feature 추가.
  * 생성된 대량의 데이터를 `data/{SYSTEM}/` 폴더에 `.npz` 포맷으로 저장/로드하여 재사용성 확보.

### 2. 트러블슈팅 (Troubleshooting)
* **이슈 1:** `ValueError: not enough values to unpack`
  * **원인:** `_generate_one_sample` 함수 인자는 4개(`dist_params` 추가)로 늘렸으나, 호출부(`args_list`)에서 3개만 넘김.
  * **해결:** `args_list` 생성 시 `self.dist_params` 추가.
* **이슈 2:** `AssertionError: Found non-positive Glucose values`
  * **원인:** 테스트 코드가 데이터의 2번째 채널(미분값, $\dot{G}$)까지 농도($G$)로 착각하여 양수 검사를 수행함. 미분값은 음수일 수 있음.
  * **해결:** 테스트 코드에서 농도(Index 0)와 미분(Index 1)을 분리하여 검사하도록 로직 수정.
* **Note (Sim-to-Real Mismatch):**
  * 현재 증강 데이터는 `(N, 5, 2)` (값 + 미분) 형태이나, 실제 NIH 데이터 로더는 `(N, 5, 1)` (값) 형태임.
  * **Action Item:** 추후 학습 단계에서 실제 데이터 로더에도 수치 미분 등을 적용하여 차원을 일치시켜야 함.