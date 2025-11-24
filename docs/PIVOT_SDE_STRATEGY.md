# 전략 전환 노트: SDE 모델링의 재정립 (Refining SDE Modeling Strategy)

**Date:** 2025-11-22
**Topic:** Correction of Diffusion Term Estimation & Introduction of Drift Bias Correction

## 1. 회고 (Retrospective): 무엇이 문제였는가?

### 1.1. 기존 접근의 오류 (The Fallacy)
우리는 SDE의 확산 계수(Diffusion Coefficient) $\sigma(t)$를 추정하기 위해, 특정 시점 $t$에서의 **상태 변수 잔차(State Residual)의 분산**을 사용했다.
$$\sigma_{old}(t) \approx \sqrt{\text{Var}(Y_{real}(t) - Y_{sim}(t))}$$

이것은 **치명적인 차원 오류**이자 개념적 오해였다.
* **SDE의 Diffusion:** 단위 시간당 발생하는 변화율의 불확실성 (Process Noise). 즉, **"속도의 노이즈"**.
* **State Residual:** $0$부터 $t$까지 누적된 오차의 합 (Accumulated Error). 즉, **"위치의 오차"**.

이로 인해 $\sigma$가 과대평가되었으며, SDE 궤적은 실제 동역학의 미세한 떨림(Fluctuation)이 아니라 거대한 범위의 랜덤 워크로 발산하게 되었다.

### 1.2. 발견된 통찰 (Insight from Failure)
이전 단계의 `noise_calibration` 결과에서 잔차의 **평균($\mu$)이 0이 아님**을 확인했다.
* 이는 단순한 랜덤 노이즈가 아니라, 결정론적 수리 모델 $f(x, \theta)$가 현실의 동역학을 구조적으로 따라가지 못하는 **Model Bias (Drift Error)**가 존재함을 시사한다.
* 따라서, 우리는 노이즈($\sigma$)만 늘릴 것이 아니라, **모델의 편향($\mu$)을 보정**해야 한다.

---

## 2. 이론적 수정 (Theoretical Correction)

### 2.1. Diffusion Term 재정의: 증분의 분산 (Variance of Increments)
SDE의 수학적 정의(Quadratic Variation)에 따라, 확산 계수는 **"짧은 시간 $\Delta t$ 동안의 변화량(Increment)의 분산"**으로 정의되어야 한다.

$$\text{Var}(\Delta Y_t) \approx \sigma(t)^2 \Delta t$$

따라서, 우리는 관측 데이터의 **구간별 변화량($\Delta Y$)**을 통해 $\sigma(t)$를 역산해야 한다.
$$\sigma(t) \approx \sqrt{\frac{\text{Var}(\Delta Y_{real} - \Delta Y_{sim})}{\Delta t_{obs}}}$$
* 여기서 $\Delta Y = Y(t_{i+1}) - Y(t_i)$.
* 이 방식은 시간 간격($\Delta t$)에 대해 정규화되므로 물리적으로 타당하다.

### 2.2. Drift Correction 도입: Bias-Corrected SDE
단순히 $\sigma$만 적용하면 분포가 넓어지기만 할 뿐, 실제 데이터의 중심(Mean)으로 이동하지 않는다. 따라서 잔차의 평균 변화율을 **Drift 항에 추가적인 힘(Forcing)**으로 더해주어야 한다.

**New SDE Formulation:**
$$dX_t = \underbrace{[f(X_t, \theta) + \mu_{bias}(t)]}_{\text{Corrected Drift}} dt + \underbrace{\sigma_{corr}(t)}_{\text{Corrected Diffusion}} dW_t$$

* $\mu_{bias}(t)$: 잔차의 변화율 평균 ($\mathbb{E}[\Delta Y_{real} - \Delta Y_{sim}] / \Delta t$).

---

## 3. 검증 계획 (Verification Plan)
구현 전, 데이터 분석을 통해 다음 가정을 검증한다.
1.  **Drift Bias의 일관성:** $\mu_{bias}(t)$가 환자 간에 일관된 경향성을 보이는가? (분산이 너무 크지 않은가?)
2.  **State Dependency:** 증분 잔차($\Delta R$)가 현재 상태 값($Y_t$)에 의존적인가? (상관관계 확인)