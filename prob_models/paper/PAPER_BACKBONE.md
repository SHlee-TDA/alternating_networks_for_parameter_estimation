# Paper Backbone — Probabilistic Decoupling for Nonlinear Inverse Problems

> **이 문서의 목적.** `prob_models` 논문(레포 내부 명칭: *Paper 2*)의 주장·방법·기여·실험을
> 한 곳에 고정한 **단일 참조 문서**다. 논문의 어떤 주장을 근거로 삼아야 할 때 프로젝트 전체를
> 뒤지지 말고 이 문서를 먼저 본다. 소스(`.tex`)와 이 문서가 어긋나면, 그것은 **수정해야 할
> 불일치**이지 "둘 중 하나가 맞는" 상황이 아니다. 갱신 시 날짜와 근거를 남긴다.
>
> 최초 작성: 2026-07-09. 근거: `prob_models/paper/{iclr2026/main.tex, sections/*.tex}`,
> 구현 `prob_models/{models,trainer,infer,analysis,metrics,config,main}.py`,
> 시스템 `systems/ogtt_simul.py`, 그림 스크립트 `figure3.py`.

---

## 0. 한 줄 논지 (Thesis)

희소·부분 관측 하의 비선형 ODE 파라미터 추정에서, **결정론적** 분리 반복망은 (a) 전역 수축
(spectral normalization, `L<1`)을 강제하느라 표현력이 병목되고, (b) **구조적 비식별(structurally
unidentifiable)** 영역에서 MSE를 최소화하면 조건부 평균으로 붕괴(regression-to-the-mean = *model
collapse*)한다. 본 논문은 분리 철학을 **생성 공간**으로 들어올려 이 두 문제를 동시에 푼다:
**Dual CVAE + (추계적) Pseudo-Gibbs 샘플링**, 그리고 전역 수축 대신 **조건 노이즈 주입에 의한
국소 Jacobian 정규화**.

방법 명칭(연구노트): **A-DCVAE (Alternating Denoising CVAEs)**. 세 가지 이슈를 세 가지 해법으로
대응하는 구조가 논문의 논증 골격이다(§3.4 참조):
- Issue 1 (불안정: ε-비호환 + 오차 증폭) → **Thm 1** 조건 노이즈 = 국소 Tikhonov → 소산성/수렴.
- Issue 2 (틀린 방향: 잘못된 평균 수렴) → **Thm 2** 타깃 노이즈 = denoising score matching → 방향.
- Issue 3 (mode collapse) → **Thm 3** 추계적 PGS(무한 혼합 커널) → ergodicity + 붕괴 우회.

---

## 1. 문제 설정 (Problem Formulation)

- 시스템: `ẋ(t) = f(x(t); θ)`, 상태 `x∈R^d`, 시간불변 파라미터 `θ∈R^p`.
- 두 가지 병목:
  1. **부분 관측(비식별)**: `x = [x_obs; x_hid]`. OGTT에서는 혈당 `G`만 관측(`x_obs`),
     인슐린 `I`은 은닉(`x_hid`). 서로 다른 `θ`가 동일한 관측 궤적을 낼 수 있음 → 비식별.
  2. **희소 샘플링(temporal aliasing)**: `t∈{t_0,…,t_N}`에서만 관측. 수치 미분 기반 고전
     파이프라인이 불안정.
- 목표: 파라미터의 사후분포 `p(θ | X_obs)`를 특성화. 은닉 궤적이 완전히 결측이라 직접 최적화는
  ill-posed → **결합 사후** `p(θ, X_hid | X_obs)`를 모델링하고 교대 샘플링으로 근사.
- 두 조건부 생성기로 분해:
  - Hidden Trajectory Generator: `p_φ(X_hid | X_obs, θ)`
  - Parameter Generator: `p_ψ(θ | X_obs, X_hid)`
- **Amortized inference**: 합성 데이터 `D = {(X_obs, X_hid, θ)}`(물리적으로 타당한 prior에서
  ODE 순전파로 생성)로 오프라인 학습 → 신규 관측에 즉시 추론.

## 2. 동기와 선행 연구의 한계 (Motivation)

### 2.1 왜 중요하고 왜 어려운가
- **중요성**: 희소·부분 관측 하 역학계 파라미터 추정은 생물학·생리학 등에서 핵심 과제.
- **문제의 본질(4중고)**:
  1. Parameter estimation 자체가 불량조건 역문제(ill-posed).
  2. **Sparse observation** → ill-conditioning 극대화.
  3. **Partial observation** → 여러 θ 조합이 동일 관측 → **비식별(unidentifiability)**.
  4. **ODE simulation** → 시뮬레이션-현실 간극(Sim2Real gap).

### 2.2 기존 접근의 한계
- **전통적 기법**(NLLS, adjoint, MCMC): 고차원 비볼록 손실 지형에서 최적화 난망.
- **단일 신경망** `θ̂=f(x_obs)`: 고분산·과적합, 동역학을 이해하기보다 데이터를 암기.
- **PINN**: 샘플마다 재학습 → 추론 비효율.
- **결정론적 반복망(선행 연구)**:
  - decoupling `x_obs↦θ`를 `x_obs↦x_hid↦θ`로 분리, `H_φ(x_obs,θ)=x_hid`, `P_ψ(x_obs,x_hid)=θ`.
  - 반복 `θ_{t+1}=T(θ_t;x_obs)=P_ψ(x_obs,H_φ(x_obs,θ_t))`.
  - SN으로 `T`를 수축사상화 + teacher forcing으로 고정점 `θ*`를 정답에 정렬.
  - **한계 (두 실패 모드)**: (1) SN이 표현력을 과도 제한 → 고편향 모델. (2) 비식별 시 모든 해의
    평균점으로 붕괴(regression-to-mean / mode collapse).
- 레포 내 교차 근거: 결정론 논문 게이트 `docs/experiment-plan.md` G0에서, 학습된 반복 연산자
  `T`의 참값 `θ*`가 **끌어당기는 고정점이 아님**(≈5% 오차 바닥)이 문서화됨. *(주의: 그 게이트
  문서/실험은 결정론 논문의 것이며 이 논문의 실험 계획이 아니다. §7 참조.)*

### 2.3 아이디어 (패러다임 전환)
- **확률론적 모델**: 단일 점추정이 아니라 사후분포를 근사 → 비식별성을 **불확실성 공간**으로
  투영해 표현. 불확실성 정량화는 의료·생물 응용에서 큰 이점.
- **Pseudo-Gibbs 대응**: 결정론의 '반복 업데이트' ↔ 확률론의 'Gibbs 샘플링'. 조건부 `P(X|Y)`,
  `P(Y|X)`에서 교대 샘플링하면 결합 `P(X,Y)` 획득. Heckerman et al.(2000)의 dependency
  network용 pseudo-Gibbs 차용.
- **생성 모델**: 다봉 조건부를 표현하기 위해 두 CVAE 채택 —
  `H_φ≈P(x_hid|x_obs,θ)`, `P_ψ≈P(θ|x_obs,x_hid)`.

## 3. 제안 방법 (Method)

### 3.1 아키텍처 — Dual CVAE
- **Hidden-State CVAE `H_φ`**: 잠재 `z_H~N(0,I)`. 인코더 `q_φ(z_H|X_obs,X_hid,θ)`, 디코더
  `g_φ(X_obs,θ,z_H)`가 `X_hid`의 평균을 출력(구현상 마지막 `Sigmoid`).
- **Parameter CVAE `P_ψ`**: 잠재 `z_P~N(0,I)`. 디코더 `g_ψ(X_obs,X_hid,z_P)`가 `θ`의 평균을
  출력(구현상 마지막 `Tanh`).
- **Single CVAE (baseline)**: `x_obs → θ`만 직접 매핑(은닉 상태 무시). 디코더 마지막 `Tanh`.
- 구현: `prob_models/models.py`. 백본은 LayerNorm+SiLU Residual MLP(`build_mlp`).

### 3.2 학습 목적 — 이중 노이즈 주입 조건부 Denoising ELBO
두 네트워크를 **Conditional Denoising Autoencoder**로 학습. 조건(condition)과 타깃(target)
양쪽에 가우시안 노이즈를 주입하되 **깨끗한 타깃을 복원**하도록 강제.
- **Condition noise** `ε_c`: `θ̌=θ+ε_c`, `X̌_hid=X_hid+ε_c` → (Thm 1) 국소 수축 유도.
- **Target noise** `ε_t`: `X̃_hid=X_hid+ε_t`, `θ̃=θ+ε_t` → (Thm 2) score matching 유도.
- 손실: 노이즈가 낀 ELBO = 재구성(MSE, 깨끗한 타깃) + `β·KL`. `β`-annealing 웜업으로 posterior
  collapse 방지.
- 구현 대응(`trainer.py`): H-net은 인코더에 `y_target_noisy`+조건 `p_cond_noisy`를 넣고 clean
  `y`를 복원. P-net은 인코더에 `p_target_noisy`+조건 `y_cond_noisy`를 넣고 clean `p`를 복원.
  노이즈 스케일 config 키: `CONDITION_NOISE_STD_{Y,P}`, `TARGET_NOISE_STD_{Y,P}`.
  - ⚠️ **불일치 주의**: `config.py`는 `CONDITION_NOISE_STD_*`/`TARGET_NOISE_STD_*`를 정의하나
    `trainer.py`는 `COND_NOISE_STD_*`/`TARGET_NOISE_STD_*`를 `getattr(default=0.05)`로 읽는다.
    `COND_*` 키는 config에 없으므로 **조건 노이즈가 항상 기본값 0.05로 고정**된다(설정이
    실제로 반영되지 않음). §8 이슈 목록 참조.

### 3.3 추론 — Noise-Injected Pseudo-Gibbs Sampling (PGS)
- 결정론 디코더 출력에 스케일된 가우시안 노이즈를 명시적으로 더하고 잠재에 대해 주변화 →
  **무한 가우시안 혼합(infinite mixture of Gaussians)** 전이핵. 다봉·비식별 valley를 표현할
  무한 용량 확보.
- 교대 업데이트(iteration `k`):
  - `z_H~N(0,I)`; `X_hid^(k+1) = g_φ(X_obs,θ^(k),z_H) + ε_H`, `ε_H~N(0,σ_H²I)`
  - `z_P~N(0,I)`; `θ^(k+1) = g_ψ(X_obs,X_hid^(k+1),z_P) + ε_P`, `ε_P~N(0,σ_P²I)`
  - (선택) 물리적 경계 사영 `B_y, B_p`(clamp).
- 전이핵 `K((X_hid,θ)→(X_hid',θ')) = p(X_hid'|X_obs,θ)·p(θ'|X_obs,X_hid')`.
- 여러 체인을 병렬로 돌리고 burn-in `B` 폐기. 구현: `infer.py::pseudo_gibbs_sampling`
  (config: `INFERENCE_CHAINS/STEPS/BURN_IN`, `INFER_NOISE_{Y,P}`, temperature `τ_y,τ_p`).
- 근거 문헌 포지셔닝: Heckerman et al. (2000)의 pseudo-Gibbs는 **이산** 도메인에 한정 →
  본 논문은 연속·비식별 도메인으로 확장하되, 결정론망을 그대로 쓰면 평균 붕괴하므로 노이즈
  주입으로 해결한다는 것이 핵심 주장.

## 4. 이론적 주장 (Theorems) — 연구노트 반영 최종본

세 정리는 §3.4의 **3 이슈 → 3 해법** 구조와 1:1 대응한다. 증명 위치: `sections/appendix.tex`(활성).
비활성 중복 초안 `sections/theorems.tex`는 삭제/병합 대상. 아래 진술은 연구노트(2026-07-09)의
방어 가능한 형태를 **정본**으로 채택한다(이전 `L<1` 직접 주장은 폐기).

**문제 배경 — 세 가지 이슈 (연속·비식별 공간의 pseudo-Gibbs가 겪는 병목)**
- **Issue 1 (불안정 / 오차 증폭)**: 완벽한 데이터로 독립 학습해도 두 CVAE는 단일 결합분포에서
  도출 불가능한 **ε-비호환(ε-incompatibility)** 상태. 유효 궤적 매니폴드 `M`은 르베그 측도 0의
  얇은 공간이라, 안전장치 없는 전이는 한 스텝의 작은 오차도 피드백으로 **지수 증폭·발산**.
- **Issue 2 (틀린 방향)**: 수렴하더라도 비식별계에서 잘못된 평균점으로 수렴할 수 있음.
- **Issue 3 (mode collapse)**: 디코더 평균만 주고받는 결정론 핑퐁은 regression-to-the-mean 필연.

**정리 (정본 진술)**
- **Thm 1 (Condition Noise ⇒ Local Jacobian / Tikhonov Regularization).** 조건 변수에
  `ε_c~N(0,σ_c²I)`를 주입해 타깃을 예측하는 ELBO-MSE 목적은 `σ_c²→0`에서
  `E_ε‖H(θ+ε)−Y‖² ≈ ‖H(θ)−Y‖² + σ_c²‖∇_θ H(θ)‖_F²`로 근사. 즉 **평균 예측기 Jacobian의
  Frobenius norm을 억제하는 국소 Tikhonov 정규화와 동치**. 이는 출력 발산을 막고 시스템에 강한
  **소산성(dissipativity / Lyapunov drift)**을 부여하여 마르코프 체인이 발산하지 않고 안정
  수렴하게 한다. → **Issue 1 해결(Convergence/안정성)**.
  - 엄밀성: `L<1`을 직접 주장하지 **않고** "Jacobian 억제 → 소산성"으로 진술 → 방어 가능.
    부록의 옛 "`L≤‖∇g‖_F<1`" 문장은 삭제/약화 필요(W4).
- **Thm 2 (Target Noise ⇒ Denoising Score Matching).** 타깃에 `ξ~N(0,σ_t²I)`를 주입하고 clean
  타깃을 복원하도록 학습한 conditional DAE의 전역 최적 출력은
  `R*(Ỹ) ≈ Ỹ + σ_t²∇_Y log p(Y|X_obs,θ)` (Tweedie). 즉 복원 연산 = 주변화 분포 score를 향한
  **Langevin drift**와 동치 → off-manifold 입력을 고밀도 궤도로 되끌어오는 **인력장(attractor
  field)** 형성. → **Issue 2 해결(Correctness/방향)**.
  - 엄밀성: Vincent(2011)·Tweedie 표준 결과. 견고. 부록 표기 오류만 정리.
- **Thm 3 (Ergodicity & Score-Aligned Convergence via Infinite Mixture Kernel).** 매 스텝 `z`를
  새로 샘플링해 형성되는 **무한 가우시안 혼합** 전이핵 `T`는, Thm 1의 소산성 하에서
  **2-Wasserstein 공간에서 기하적 ergodicity**를 확보하여 유일 정류분포 `π*`로 수렴. 커널 자체가
  다봉(U자 valley)을 담을 표현 용량을 가지고, Thm 2의 Langevin drift가 지속 인력을 제공 → 체인이
  mode collapse를 우회하고 참 결합사후 `p_data`에 정렬. → **Issue 3 해결(붕괴 우회)**.
  - 엄밀성: 2-Wasserstein 기하 ergodicity + 무한혼합 표현력 + score drift 정렬로 프레이밍.
    Meyn & Tweedie 드리프트 원용. Lyapunov 함수·`π*`≈참사후 연결의 세부는 부록에서 보강.

> **정본 요약**: (Thm1) 안정 → (Thm2) 방향 → (Thm3) 붕괴 우회. Thm 1의 과주장(`L<1`)은
> 폐기하고 소산성/Tikhonov로 진술. 이 형태가 writing·appendix의 기준.

## 5. 기여 & 신규성 (Contributions & Novelty)

1. 물리적 분할(은닉 상태 vs 파라미터)을 유지하면서 **표현적 불확실성 모델링**이 가능한
   **생성적 분리 프레임워크**(dual-network).
2. **조건 노이즈 주입 = Jacobian에 대한 암묵적 Tikhonov 정규화** → 국소 Lipschitz 제어 →
   안정적 경험적 MCMC 수렴. (전역 spectral 제약을 국소 정규화로 대체.)
3. 비식별 매니폴드를 따라 **다봉·구조화된 사후(bimodal)** 를 복원 → 결정론 회귀의
   regression-to-the-mean 실패를 우회.

신규성의 핵심은 (기존) 결정론 분리·전역수축 → (본 논문) 생성 분리·국소정규화·연속 pseudo-Gibbs
로의 전환. "이산 pseudo-Gibbs → 연속·비식별" 확장과 "노이즈 주입의 이중 역할(수축+score)"이
차별점.

## 6. 실험 주장 & 필요한 그림/표 (Experimental Claims — 그림 대응표)

시스템: OGTT 최소모델(`systems/ogtt_simul.py`). 파라미터 `θ=(S_I, σ)`(인슐린 감수성, 분비
용량). 관측=혈당 `G`(`observed_var_idx=0`), 은닉=인슐린 `I`(`hidden_var_idx=1`). `t={0,30,60,90,120}`.

| 그림/표 | 무엇을 보여야 하나 | 데이터 출처(코드) | 상태 |
|---|---|---|---|
| **Fig 1 (Teaser)** | Single net은 unstructured blob으로 붕괴 / Dual CVAE는 초승달 valley 추적 | 개념도(+ 실제 산점 인셋 가능) | ❌ placeholder(`\fbox`) |
| **Fig 2 (MCMC diag)** | trace plot 수렴/mixing + ACF 빠른 감쇠 | `analysis.py::plot_mcmc_trace_and_acf`, `infer.py`의 `theta_history` | ❌ placeholder |
| **Fig 3 (Posterior)** | (a) 전역 joint `p(θ)`는 둘 다 커버 (b) 국소 `q(θ|x_obs)`: Single은 mode collapse(참값 miss), Iter는 비식별 곡선 `S_I·σ=C` 따라 stretch | `figure3.py` (B/C 패널) | 🟡 **PDF 2개 존재**(`figures/figure3_{B,C}_*.pdf`); A(1D) 미포함, 재현 스크립트 있음 |
| **Fig 4 (Predictive check)** | bimodal 두 봉우리에서 뽑은 θ로 순전파 → 둘 다 희소 관측 재구성 | 순전파(OGTT `simulate`) + posterior 샘플 | ❌ placeholder |
| **Fig 5 (Noise sensitivity)** | 결정론=평탄(고편향), 생성=off-manifold에서 분산 스파이크 | `analysis.py::evaluate_robustness_probabilistic` → `robustness_{ours,baseline}.csv` | ❌ placeholder(CSV 생성 코드는 있음) |
| **Table 1 (정량)** | 점추정(RMSE/Pearson) + 보정(PICP/MPIW/CRPS/NLL) | `analysis.py::run_prob_evaluation_phase`, `metrics.py` | ✅ 값 채워짐(`experiment1.tex`) |

**Table 1 현재 값**(`experiment1.tex`): Single CVAE는 RMSE(0.145/0.148)·Pearson이 근소 우위지만
PICP 17.2%, NLL 31990.99로 파탄. Iter CVAEs는 RMSE 근소 열세지만 PICP 99.44%, NLL −0.808.
서사: RMSE만 보는 것은 비식별계에서 "정확성의 착시". ⚠️ MPIW가 0.0102→1.1466로 크게 넓어짐 —
"과대 불확실성(under-confidence)" 반론 가능. §8 참조.

**핵심 비식별 주장(✅ 증명 확보, 2026-07-09)**: 국소 사후가 하이퍼볼라 `S_I·σ=C`(=`mDI` 등온선)를
따른다는 것은 **collaborator(Ha Joon, Howard) supplement §II 비차원화 증명**으로 확립됨: 혈당 `G(t)`는
`mDI=S_I·σ`에만 의존, 인슐린 `I(t)`는 `S_I` 배수 → 혈당만 관측 시 `S_I,σ` 구조적 비식별. `figure3.py`의
`C=S_I,true·σ,true` fiber는 증명과 **정확히 일치**(수정 불필요). 이 사실은 (1) decoupling 동기를
물리적으로 grounding하고(은닉 인슐린이 곧 비식별을 푸는 정보), (2) `p(S_I,σ|G)∝L(S_I·σ)·prior`로
**해석적 참조 posterior**를 값싸게 제공한다(B3). 상세: [DISCUSSION.md](DISCUSSION.md) B2/B3.

## 7. 범위 주의 — 이 논문 ≠ `docs/experiment-plan.md`

`docs/experiment-plan.md`의 게이트(G0–G2)와 실험(E1–E6), 그리고 `CLAUDE.md`의 5-Step Rule은
**결정론(spectral normalization) 논문**의 실험 프로그램이다. 본 세션의 `prob_models` 논문은
별개(Paper 2)이며, 위 계획서의 절차를 그대로 상속하지 않는다. 다만 그 계획서의 **아이디어**
(비식별 valley 정량화 E5, 노이즈 sweep E1)는 본 논문 Fig 3·5의 근거로 재활용 가치가 있다.

## 8. 발견한 이슈 / 리스크 (작성/실험 전에 처리)

**코드**
- (C1) **조건 노이즈 설정 무효화**: `trainer.py`가 `COND_NOISE_STD_{Y,P}`를 읽는데 `config.py`엔
  `CONDITION_NOISE_STD_{Y,P}`만 있음 → 조건 노이즈가 항상 0.05 고정. **[결정 2026-07-09]**
  의도적으로 남겨둠(고정 조건노이즈 0.05로도 잘 되는지 먼저 검증, 성공적 판단). 키 통일은
  **본 실험(condition-noise ablation 등) 수행 시점에** 처리하기로 연기.
- (C2) `SingleCVAE`/`ParameterCVAE` 디코더 마지막 **`Tanh`가 정규화 파라미터 공간을 [-1,1]로
  캡**. 정규화 스킴에 따라 값 절단 위험(레포 메모리의 결정론쪽 Tanh-cap 버그와 동종). 확인 요망.
- (C3) `HiddenStateCVAE` 디코더 마지막 `Sigmoid` → 은닉 상태가 [0,1] 정규화 가정. 정규화기와
  정합 확인.
- (C4) `infer.py`는 `infer_noise_y/p`, `latent_dim`을 모델 속성에서 `getattr`로 읽음. `config`의
  `INFER_NOISE_*`가 모델에 주입되는 경로 확인(현재 기본 0.05 fallback 가능).

**논문/서술**
- (W1) ✅ **해결(2026-07-09)**: `04_method.tex`의 깨진 ELBO 수식을 `\gL_H,\gL_P` 정본으로 재작성.
- (W2) ✅ **해결(2026-07-09)**: 활성 4개 파일(main/prob_formulation/04_method/appendix/experiment1)
  표기 통일 — `\vtheta, \mX_{\mathrm{obs}}, \mX_{\mathrm{hid}}, \vz_H/\vz_P, \boldsymbol{\epsilon}`.
  잔재 grep 무결 확인. (dlbook 매크로: `\vzero,\mI,\Tr,\E,\gL,\gH,\gP` 사용.)
- (W3) ✅ **해결(2026-07-09)**: 비활성 중복 초안 `sections/{method,train,theorems,infer,
  01_introduction}.tex` 삭제. `main.tex`의 주석 처리된 `\input` 라인도 함께 제거.
- (W4) ✅ **해결(2026-07-14)**: Thm 1 `L<1`→"Jacobian 억제/소산성" 약화(2026-07-09) + PICP/MPIW
  트레이드오프 반론을 `experiment1.tex` "Coverage vs sharpness" 문단으로 서사 보강(방향성 검증=
  따라-fiber vs 가로-fiber, 참조 posterior 대비는 Track E placeholder). Limitations 문단(conclusion)에
  Sim2Real 완화(B10)·실패모드(B11) 추가.
- (W5) ⚠️ **신규(2026-07-14)**: intro/abstract/background/contributions "공유 연산자 T" 서사로
  재작성(B1), Related Work 신규(B8). bib 12개 추가는 **서지 검증 필요**. 미정의 매크로 `\xobs`
  (theory/appendix 4곳)를 preamble에 정의(컴파일 오류 수정).

**저장소/작업 위치**
- (R1) 논문 소스는 이 **worktree에 없고** 메인 작업 트리
  `/home/shlee/projects/alternating_networks_for_parameter_estimation/prob_models/paper/`에
  **커밋되지 않은(staged)** 상태로 존재. 이 문서도 거기에 저장했다. 작업 브랜치/커밋 전략을
  세션 초반에 결정해야 함.

## 9. 표기·기호 빠른 참조

| 기호 | 의미 |
|---|---|
| `X_obs, X_hid` | 관측/은닉 상태 궤적 (OGTT: 혈당 `G` / 인슐린 `I`) |
| `θ=(S_I,σ)` | 파라미터: 인슐린 감수성 `S_I`, 분비용량 `σ` |
| `H_φ, P_ψ` | Hidden-State CVAE, Parameter CVAE (디코더 `g_φ,g_ψ`) |
| `z_H,z_P` | 각 CVAE 잠재 (`N(0,I)`) |
| `ε_c, ε_t` | 조건 노이즈, 타깃 노이즈 (`σ_c,σ_t`) |
| `ε_H,ε_P / σ_H,σ_P` | 추론 시 디코더 출력 주입 노이즈 |
| `K` | pseudo-Gibbs 전이핵 (또는 스텝 수 `K`, 문맥주의) |
| PICP/MPIW/CRPS/NLL | 보정 지표(`metrics.py`) |

---

### 갱신 로그
- 2026-07-09 — 최초 작성. 소스·구현·시스템 정독 기반. 이슈 C1–C4, W1–W4, R1 기록.
- 2026-07-09 — 연구노트 반영: A-DCVAE 명명, §2 동기 확장(4중고·기존기법 한계·패러다임 전환),
  §3.4 3이슈→3해법 구조, §4 정리 정본화(Thm1 `L<1`→소산성/Tikhonov로 약화, Thm3 2-Wasserstein
  ergodicity). C1은 본 실험 시점으로 연기(사용자 결정).
- 2026-07-09 — 표기/LaTeX 정리 + writing: 활성 4섹션(prob_formulation·04_method·appendix·intro/
  background) 재작성으로 표기 통일 + 깨진 ELBO 수정 + 정리/증명 정본화(A-DCVAE, 3이슈→3정리).
  bib에 11개 참고문헌 추가(heckerman/vincent/tweedie/meyn 등). 정적 검사 통과(인용키·환경짝·
  ref/label 정합·표기 잔재 0). LaTeX 툴체인 부재로 실제 컴파일은 미검증. W1/W2 해결, W4 부분해결.
