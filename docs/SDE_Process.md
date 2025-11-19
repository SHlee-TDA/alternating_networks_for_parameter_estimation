# SDE 기반 확률적 데이터 증강 프로세스 개요

## 목표: Sim-to-Real Gap 해소 (Support Mismatch 해결)

기존 결정론적 ODE 모델의 좁은 데이터 분포 ($P_{det}$)를, 실제 NIH 데이터의 불확실성을 포괄하는 확률적 튜브 분포 ($P_{sde}$)로 확장하여, 현실 데이터에 최적화된 파라미터 추정 모델을 학습합니다.

| 단계 (Phase) | 역할 (Role) | 구현 상태 (Code Status) | 담당 모듈 및 다음 계획 |
| :---: | :--- | :---: | :--- |
| **Phase 1** | **문제 정의 및 기반 마련** | **완료** | `systems/base_system.py`에 `drift_func` 및 `diffusion_func` 인터페이스 정의 완료. |
| **Phase 2** | **Noise Calibration** | **완료** | 실제 NIH 데이터를 로드하여 결정론적 궤적과 데이터 간의 잔차를 분석, $\sigma_{emp}$를 추정함. |
| **Phase 3** | **SDE 모델링 및 Solver 구현** | **완료** | Clamping 적용된 Euler-Maruyama Solver 구현 완료. |
| **Phase 4** | **데이터 증강 (Data Augmentation)** | **완료 (Verified)** | **`data_loader.py`**: SDE Solver 통합, Rejection Sampling, `.npz` 저장/로드 구현 완료. **`distribution_analysis.py`**: 데이터 분포 분석 도구 구현. |
| **Phase 5** | **모델 학습 및 검증** | **대기 중** | 확장된 데이터셋($P_{sde}$)으로 학습 수행 및 Wasserstein Distance 검증. |

---

## Phase 2 상세: Noise Calibration (현재 완료 단계)

| 코드 | 역할 상세 | 산출물 |
| :---: | :--- | :--- |
| `data_loader.RealOGTTDataLoader` | NIH 데이터를 `(Glucose: Observed, Insulin: Hidden)` 형태로 로드. **정책:** 15분 시점 제외, 결측치 샘플 삭제. `DataGenerator`와 동일한 튜플 구조 반환. | `X_obs`, `Y_hid`, `P_true`, `t_points` |
| `noise_calibration.py` | `P_true`를 Ground Truth로 간주하고 결정론적 OGTT 모델을 시뮬레이션. 실제 관측값 (`X_obs`, `Y_hid`)과의 잔차를 계산하여 노이즈의 경험적 표준편차 ($\sigma_{emp}$)를 산출. | $\sigma_{emp}^{Glucose}$, $\sigma_{emp}^{Insulin}$ |

## Phase 3 상세: SDE Core Implementation
* **시간 의존성:** 확산 계수 $\sigma(t)$는 관측 시점 사이를 선형 보간(Linear Interpolation)하여 연속성을 보장함.
* **수치 안정성:** 발산을 막기 위해 $Y_t \in [10^{-6}, 1.1 \times Y_{max}]$ 범위로 매 스텝 Clamping 적용.