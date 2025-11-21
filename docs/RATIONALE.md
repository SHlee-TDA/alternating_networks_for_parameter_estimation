# 설계 근거 및 의사결정 노트 (Design Rationale)

## 2025-11-20: 실제 데이터의 미분값 추출 전략 (Spline Smoothing)

### 1. 배경 (Context)
* **문제:** SDE 기반 증강 데이터($P_{sde}$)는 Lagrangian Feature(Drift Term, $\dot{y}$)를 포함하여 `(N, 5, 2)` 형태를 가지나, 실제 데이터($P_{real}$)는 관측값만 있어 `(N, 5, 1)` 형태임. 학습 시 입력 차원 불일치 발생.
* **목표:** 실제 데이터에서도 노이즈를 억제하면서 합리적인 미분값($\dot{y}$)을 추정하여 차원을 일치시켜야 함.
* **제약:** 데이터 포인트가 5개($t=0, 30, 60, 90, 120$)로 매우 적고, 측정 노이즈가 포함되어 있음.

### 2. 결정 (Decision)
* **방법:** **Reinsch의 평활 스플라인 (Smoothing Spline)** 사용 (`scipy.interpolate.UnivariateSpline`).
* **평활화 파라미터($s$) 설정:** Phase 2의 Noise Calibration에서 얻은 경험적 오차($\sigma_{emp}$)를 기반으로 설정.
  $$s \approx \sum_{t} \sigma_{emp}(t)^2$$

### 3. 근거 (Rationale)
1.  **Reinsch의 기준:** 통계적으로 관측값이 $y_i = f(t_i) + \epsilon_i$이고 $\epsilon_i \sim N(0, \sigma^2)$일 때, 잔차 제곱합($\sum(y_i - \hat{f})^2$)이 $N\sigma^2$ 수준이 되도록 평활화하는 것이 신호와 노이즈를 분리하는 표준적인 기준임.
2.  **Proxy로서의 $\sigma_{emp}$:** 실제 측정 장비의 $\sigma$를 모르기 때문에, 결정론적 모델과의 잔차 표준편차($\sigma_{emp}$)를 "모델이 설명하지 못하는 불확실성의 총량"으로 간주하여 대용(Proxy)함.
3.  **일관성 (Consistency):** 개별 샘플마다 $s$를 튜닝하는 것은 과적합(Overfitting) 위험이 있고 추론 시 불가능함. 데이터셋 전체의 통계량($\sigma_{emp}$)을 적용하는 것이 강건함(Robustness)을 보장함.

### 4. 잠재적 위험 및 논의 (Risks & Discussion)
* **우려 사항:** $\sigma_{emp}$가 상당히 큼(Glucose의 경우 $\approx 10 \sim 20$). 따라서 $s$값이 커져 스플라인이 데이터를 느슨하게 따라가며, $N=5$인 경우 거의 **직선(Linear)이나 완만한 2차 곡선**에 가까워질 수 있음.
* **대응 논리:** 미분값(기울기)을 Feature로 사용할 때, 노이즈에 의해 기울기가 급격히 변하는 것(Wiggling)보다는 **다소 밋밋하더라도 전체적인 증감 경향성(Trend)만 잡는 것이 학습에 더 유리**할 것으로 판단됨.****