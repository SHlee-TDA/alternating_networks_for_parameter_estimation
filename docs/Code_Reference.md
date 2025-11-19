# 코드 레퍼런스 (Code Reference) - SDE 패치 영역

## 1. data_loader.py
### `class RealOGTTDataLoader`
NIH OGTT 데이터셋을 로드하고 전처리하는 전용 클래스.
- **입력:** CSV 파일 경로, `Config` 객체.
- **전처리 정책:**
    - **시간 축:** `t=[0, 30, 60, 90, 120]` 5개 시점만 사용 (15분 제외).
    - **결측치:** 결측치(NaN)가 포함된 행은 데이터 품질을 위해 삭제(`dropna()`).
- **출력 (Return):** `DataGenerator.generate_data()`와 일관된 4-tuple.
    - `Observed Data (Glucose)`: `(N, 5, 1)`
    - `Hidden Data (Insulin)`: `(N, 5, 1)`
    - `Parameters (si, sigma)`: `(N, 2)`
    - `Time Points`: `(5,)`

## 2. noise_calibration.py
### `def calibrate_noise()`
SDE 확산 항($\sigma_{emp}$)을 결정하기 위해 실제 데이터의 불확실성 규모를 정량화하는 스크립트.
- **로직:**
    1. `RealOGTTDataLoader`로 NIH 데이터를 로드.
    2. 각 환자의 Ground Truth 파라미터로 결정론적 OGTT 시뮬레이션 실행.
    3. `Residual = Real Data - Simulated Data`를 계산.
    4. 모든 잔차의 표준편차를 $\sigma_{emp}$로 산출하여, SDE 확산 모델의 인풋으로 사용될 상수를 결정.
- **결과:** Glucose 및 Insulin에 대한 스칼라 $\sigma_{emp}$ 값.

## 3. systems/base_system.py (확장)
- **`def drift_func(t, y, params)`:** SDE의 결정론적 부분. (기존 `ode_func`와 동일)
- **`def diffusion_func(t, y, params)`:** SDE의 확률적 부분. **(SDE 구현 시 이 메서드를 오버라이딩해야 함)**

## 4. systems/ogtt_simul.py (수정)
- **`def calculate_ogtt_flux(self, t)`:** `scipy.solve_ivp`의 벡터 입력에 대응하기 위해 `if/elif` 구조를 `np.select` 기반의 벡터화 연산으로 수정.
- **`def simulate(self, ..., t_eval=None)`:** `t_eval` 인자 체크 시 `if t_eval == None`을 `if t_eval is None`으로 수정하여 NumPy 배열과의 비교 에러 방지.