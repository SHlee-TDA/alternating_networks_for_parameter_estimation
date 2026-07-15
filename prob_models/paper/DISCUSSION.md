# Research Discussion — Open Questions & Strategic Assessment (A-DCVAE)

> 이 문서는 논문을 "완성"하기 전에 함께 논의하며 채워갈 **연구 공백 목록 + 전략 판단**이다.
> `PAPER_BACKBONE.md`가 "지금 주장하는 것"을 고정한다면, 이 문서는 "아직 정당화되지 않은 것,
> 결정해야 할 것"을 추적한다. 각 항목은 상태 `[ ] open / [~] 논의중 / [x] 해결`로 관리.
> 최초 작성 2026-07-09.

---

## A. 전략 판단 요약 (지도교수 코멘트)

1. **의존성 리스크 (Q1).** 현재 draft는 "our prior work(결정론 논문)"에 서사가 종속돼 있다.
   그 논문은 (a) 미출판, (b) 자체 결함(θ* non-attracting), (c) 정직한 주제가 *"robustness이지
   정확도 아님"* 이라, 이 논문이 그것을 "실패(collapse)"라 부르는 순간 **두 논문이 서로를
   약화**시킨다. → 이 논문은 **self-contained**로 재구성해야 한다. 결정론 방법은 "우리 선행연구"가
   아니라 "결정론적 amortized 추정기라는 baseline 부류"로 격하하고, **이 논문 안에서 직접 구현·실행**
   해 collapse를 인용이 아니라 증거로 보인다.
2. **Novelty (Q2).** 개별 부품은 모두 기존 것: 노이즈=Tikhonov(Bishop 1995), denoising=score
   matching(Vincent 2011), pseudo-Gibbs=Heckerman 2000, amortized posterior=SBI/NPE. **신규성은
   합성(synthesis)과 적용**(비식별 ODE 역문제에서 "안정+방향+비붕괴"를 한 틀로 설명)에 있다.
   VAE 백본은 **기여가 아니라 instantiation**으로 격하해야 손해를 줄인다.
3. **Main contribution (Q3).** 성능(metric) 프레이밍은 함정이다(Single CVAE가 RMSE 우위인 걸
   논문이 자백함). 기여는 **"비식별 다양체(구조화된 posterior)를 붕괴 없이 복원 + 그 이유를 3정리로
   설명"**에 둔다. 실험은 성능비교가 아니라 (i) 참조 posterior 대비 구조 복원, (ii) 3정리를 격리
   검증하는 ablation으로 재설계.
4. **Target venue (Q5).** 현 설계로 ICLR/NeurIPS는 novelty·baseline에서 무너진다. 현실적 1순위는
   **AISTATS 또는 TMLR**(+SBI 포지셔닝·강baseline·ablation 완비 조건). 비식별/UQ 이론을 깊게 파면
   **SIAM/ASA JUQ**도 후보(단 결정론 논문과 venue 계열 중복 주의).

---

## A'. Scope 결정 (2026-07-09 확정)
- **정체성: "비식별성이 있는 일반 동역학 역문제를 위한 방법"**. OGTT는 **여러 예시 중 하나**(그리고
  비식별이 증명된 편리한 사례)일 뿐, 이 방법을 OGTT/mDI 문제에 국한하지 않는다.
- **용어**: 본문은 **일반 용어**(non-identifiability manifold, product degeneracy `S_I·σ`)로 서술.
  `mDI`는 "이 양의 도메인 명칭"으로 **인용과 함께 부수적으로만** 도입(load-bearing 아님). 근거:
  ML 독자에게 mDI를 전면에 세우면 예제가 과도하게 specific·비재현적으로 오해될 위험.
- **함의(→ B-items)**: 일반성 방어를 위해 **비-임상 toy 예시 1개 추가 권장**(B2/B3의 toy와 통합):
  비식별이 해석적으로 명확한 저차원 ODE. "OGTT + toy 두 사례에서 동일 프레임워크"가 일반성 증거.
- **Venue 함의**: 위 정체성은 **ML 계열(AISTATS/TMLR) 지향**과 정합. 응용수학/임상 프레이밍(mDI 전면)
  은 이 정체성과 상충하므로 지양.
- 서지: Ha et al., JCEM 2025 (`ha2025disposition`) bib 등록 완료.

## A''. 이론 뼈대 결정화 (2026-07-09) — "수렴을 증명, 그다음 방향을 조종"

**철학**: 많은 생성모델이 sampling 분포의 수렴(ergodicity)을 암묵적으로 가정. 우리는 **증명한다.**
그다음, 수렴한 정적분포를 **target으로 조종**한다. 이 2단 구조가 전작(deterministic)과 정확히 평행:

| | 전작 (deterministic) | 본작 (probabilistic) |
|---|---|---|
| **존재/수렴** | spectral norm → 수축 → Banach 고정점 `θ*` | **ergodicity → 유일 불변측도 `π*`** |
| **조종(정확성)** | teacher forcing → `θ*`가 `θ*(x_obs)`에 정렬 | **denoising → `π*`가 target posterior로** |
| **잔여 편향** | `θ*` non-attracting(~5% floor, G0) | ε-비호환 gap (Q-B) |

**Theorem 재편(정본 후보)**:
- **배경(정리 아님)**: Bishop95(condition noise=Jacobian penalty), Vincent11/Tweedie(denoiser=score).
- **Theorem A (Existence/Ergodicity)**: 연속 PGS 체인이 유일 불변측도 `π*`로 수렴.
  → **완전 형식화된 backstage 노트: `theory_notes.tex`** (정의·Doeblin 도구·Lemma·Thm A 증명 완비,
    독립 컴파일 가능; 이후 정리들의 재사용 기반).
  - Heckerman00은 **이산** PGS 수렴(유한상태 → Perron-Frobenius로 자명). **연속 확장이 진짜 일**.
  - 증명 경로: **Doeblin minorization**(compact Ω + 양의 가우시안 커널 → 유일 `π*`, 기하수렴).
    ⭐ 이 경로의 가정(compact, 양의 커널)은 **우리 알고리즘이 구성으로 보장**(경계사영 + 가우시안
    노이즈) → "가정 안 하고 증명" 철학에 **정확히 부합**. 반면 Wasserstein-수축 경로는 `L<1`을
    **가정**해야 하므로(네트워크가 보장 못 함) 철학상 약함. → **Doeblin이 primary, 수축은 rate용**.
  - condition-noise(옛 Thm1)의 역할: 존재가 아니라 **수렴 rate/저노이즈 강건성/집중도**.
- **Theorem B (Correctness/Steering)**: denoising이 `π*`를 target posterior로 조종.
  - denoiser=score → 각 스텝이 데이터 다양체로의 Langevin drift → `π*`가 참 support에 집중.
  - (이상) 조건부 호환+정확 → `π*`=참 posterior. (실제) 비호환+근사오차 → **Q-B 편향 상한**.
  - 전작 teacher forcing의 확률론적 대응물.

**Doeblin 학습 필요**: student가 개념 미숙지 → §D에 직관 설명 추가. 철학상 필수 도구이므로 이해 필요.

## B. 반드시 메워야 할 연구 공백 (Discussion Items)

### B1. 서사 독립성 — 결정론 선행연구 탈종속  `[x]`  (2026-07-14 intro 작성 완료; 실험증거는 Track E 대기)
> **2026-07-14 반영**: `main.tex` abstract·intro·background·contributions 재작성. "our prior work"
> 제거→"deterministic decoupled baseline". "공유 연산자 T, 수렴 대상을 고친다" + mixture identity
> `p(θ|x_obs)=∫p(θ|x_obs,x_hid)p(x_hid|x_obs)`로 decoupling 유도. 고정점=E[θ|x_obs]는 inline 1줄
> (companion 인용 없이 자명). spectral norm/Banach/teacher-forcing 증명·G0 non-attracting은 미도입(계획대로).
> ⏳ 남음: 결정론 baseline **직접 실행** 증거(B1 할일 (a))는 Track E.
- 문제: intro/abstract가 "our prior work"에 기댐(미출판=검증불가). 그러나 regression-to-mean
  단독 유도는 "왜 **두 네트워크를 iteration**하는가"에 답하지 못함 — 그건 결정론 스토리의 '끝'.
- **해소 프레이밍 (채택): "공유 연산자 T, 무엇에 수렴하는지를 고친다".**
  - "왜 두 네트워크를 iteration하는가"의 정당화는 **결정론 machinery가 아니라 확률론-native
    논거**로 세운다(선행논문 불필요):
    1. 목표 `p(θ|x_obs)`는 부분관측 하에서 넓은 비식별 ridge.
    2. **구조적 사실**: 전체 궤적에 조건화하면 θ는 식별 가능 → `p(θ|x_obs,x_hid)`는 sharp/well-posed.
       비식별성은 전적으로 결측 `x_hid`에 있다.
    3. 따라서 `p(θ|x_obs)=∫ p(θ|x_obs,x_hid) p(x_hid|x_obs) dx_hid` = **sharp 조건부들을 x_hid
       불확실성으로 혼합**한 것. ridge는 x_hid 주변화로 생성됨.
    4. 이를 샘플링하는 자연스러운 방법 = `x_hid|x_obs,θ`와 `θ|x_obs,x_hid`를 교대 샘플링 = **Gibbs**.
       → 두 네트워크 iteration은 heuristic이 아니라 **결합사후의 sampler**. ("왜 iterate"에 답)
  - **결정론 방법과의 연결(브릿지, 상충 아님)**: 같은 연산자 `T`에서 결정론은 각 조건부의 **평균**을
    주고받아 고정점이 `E[θ|x_obs]`(ridge의 평균, 비식별 시 저밀도/비물리 "중앙")로 수렴. 확률론은
    결정론 맵을 **추계적 커널**로 대체해 **같은 iteration이 평균이 아니라 조건부 전체를 샘플**.
    → 두 논문은 상충이 아니라 **상보**(그쪽: "평균으로 수렴 + 그 평균의 robustness 가치"; 이쪽:
    "평균은 비식별 하에서 lossy, 조건부 전체를 복원"). repo 문서가 이미 "fixed point = E[θ|x_obs]"를
    명시(root CLAUDE.md).
- 할 일: (a) 결정론 iterative를 baseline으로 **직접 실행**(master_train.py가 이미 지원), collapse를
  in-paper 증거로. (b) 위 4단계 + 브릿지를 method motivation으로 서술. (c) 결정론 논문은
  **companion/concurrent 인용** + inline 1줄 유도로 처리(load-bearing 아님).
- ⚠️ **실험 caveat**: 결정론 baseline이 실패하는 이유를 **개념적 collapse(평균이 무정보)**로
  귀속해야 함. G0가 밝힌 "θ*가 attracting fixed point이 아님(~5% floor)"이라는 결정론 논문의
  **미해결 최적화 병리**를 이 논문의 증거로 오인/전시하지 말 것. 둘을 분리해 제시.
- **Import 최소집합(분량/focus 방어)**: 가져올 것 = ① 연산자 T 구조(1문단, Gibbs로 자명),
  ② "결정론 고정점 = 조건부 평균"(inline 3줄 + companion 인용), ③ baseline 실험 증거.
  **빼놓을 것** = spectral norm / Banach / teacher-forcing alignment 증명 / G0 non-attracting 논의.

### B2. 비식별 구조 (fiber = S_I·σ)  `[x]`  (2026-07-09 RESOLVED — 증명 확보)
- **증명 확보 (Ha Joon collaborator, supplement §II "Dimensional Analysis")**: 비차원화로
  `g(τ), i(τ)`가 **오직 `S_I·σ`에만 의존** → 혈당 `G(t)`는 `mDI := S_I·σ`에만 의존, 인슐린 `I(t)`는
  `S_I` 배수로 스케일. ∴ **혈당만 관측 시 `S_I, σ`는 구조적 비식별(곱으로만 등장), `mDI=S_I·σ`만
  식별 가능. 인슐린 관측 시 `I`의 `S_I`-스케일로 둘 다 식별.**
- ✅ **figure3.py 정합성 확인**: 코드가 `σ = (S_I,true·σ_true)/S_I` 즉 `S_I·σ=const`를 그림 →
  **증명과 정확히 일치**. fiber 수정 불필요. 단 "이론적 매니폴드"를 **`mDI` 등온선**으로 명명하고
  **인용**할 것.
- **정직성 note**: 증명은 축약모델(HGP의 S_I,I 의존 제거+정상상태)에서 **정확**, 완전모델에서는
  **수치적으로 G 불변 확인(Supp Fig 2)**. 논문 표현: "구조적 비식별(축약형에서 증명, 완전형에서
  수치 확인)".
- **TODO**: 서지정보(어느 논문의 supplement인지)로 bib 추가. `mDI`/`DI(disposition index)` 용어
  채택. co-authorship/credit 확인.

### B3. 참조(gold-standard) posterior  `[~]`  (2026-07-09 — OGTT는 해석적으로 near-solved)
- **핵심**: B2 증명 덕에 OGTT 참 posterior가 **값싸게 해석적으로** 특성화됨.
  `p(S_I,σ|G) ∝ L(S_I·σ)·p(S_I,σ)` — 우도가 곱 `mDI`에만 의존하므로:
  - **fiber 가로(across)**: 관측노이즈가 결정하는 두께(우도 sharpness in mDI).
  - **fiber 세로(along)**: **prior `p(S_I,σ)`를 등온선에 제한한 것**이 밀도 결정(우도는 fiber 위에서
    평평).
  → `(S_I,σ)` grid에서 forward-sim으로 `G` 만들어 `L(mDI)` 계산 + prior → **정확한 2-D 참조
    posterior**(ABC/long-MCMC 불필요). 우리 π*와 2-Wasserstein/coverage/along-fiber KL로 비교.
- 남는 일: toy 시스템(비-OGTT) 하나에서도 같은 검증을 반복해 일반성 확보(옵션). grid 우도 계산
  스크립트 작성.

### B4. 강한 baseline (decoupling이 진짜 이득인가)  `[~]`  (2026-07-14 — 10k 결론 뒤집힘, 캐노니컬 재검증 중)
- ✅ 구현: `prob_models/paper/experiments/b4_baselines.py`. 세 방법 모두 **동일 컨텍스트**
  (data/split/normalizer 동일)에서 학습, B3 참조 posterior 대비 비교(5 seed × 5 ref = 25 run/method).
  - **A-DCVAE(dual, ours)**, **NPE-flow**(self-contained conditional RealNVP 단일망 `p(θ|x_obs)`,
    외부설치 없음), **det-reg**(결정론 MSE regressor — 개념적 collapse, along-fiber std 0).
  - 결정론 iterative는 iter_det 체크포인트 대신 깨끗이 수렴하는 MSE regressor 사용 → **G0
    non-attracting 병리 전시 회피**(A''/B1 caveat 준수).
- **10k-scale 결과 (mean±std):**
  | | ref-HPD cov↑ | along-fiber std↑ | sliced-W2↓ | mDI err↓ |
  |---|---|---|---|---|
  | A-DCVAE | 0.20 | 0.88 | 0.156 | 0.022 |
  | NPE-flow | **0.81** | 1.23 | **0.148** | 0.044 |
  | det-reg | 1.0* | **0.0** | 0.29 | 0.033 |
  → 당시 결론: decoupling이 단일망 NPE를 명확히 못 이김(sliced-W2 무승부, coverage는 NPE 우위).
- **50k 캐노니컬 스케일 재현 (v1, 2026-07-14):**
  | | ref-HPD cov↑ | along-fiber std↑ | sliced-W2↓ | mDI err↓ |
  |---|---|---|---|---|
  | A-DCVAE | 0.20 | 0.71 | **0.170** | **0.016** |
  | NPE-flow | 0.61 | 1.995(과대분산) | 0.481(3배 악화) | 0.012 |
  | det-reg | 1.0* | 0.0 | 0.284 | 0.040 |
  → **순위가 뒤집힘**(A-DCVAE가 sliced-W2·방향 모두 우위)처럼 보였으나 —
- ⚠️ **핵심 confound 발견 (v1 결과 신뢰 보류)**: `train_npe_flow`/`train_det_regressor`가
  **patience=40 하드코딩**, A-DCVAE(via `B7.train_variant`)는 config 기본값 **patience=200**을 받음
  → 5배 학습예산 불균형. 이 confound는 10k·50k 양쪽에 동일하게 있었지만, 50k에서 "같은 epoch-patience가
  상대적으로 더 이른 종료"로 작용해 NPE/det-reg가 불리해졌을 가능성. **2026-07-14 수정**:
  `b4_baselines.py`에 `--patience` 인자 추가, 세 방법 모두 통일 적용. **v2(patience=200 통일) 캐노니컬
  재실행 진행중** — 완료 후 이 항목·`figure_b4_baselines.pdf`·`RESULTS.md` 갱신 예정.
- **당분간 결론 보류**: v2 완료 전까지 "decoupling이 이긴다/못 이긴다" 어느 쪽도 논문에 확정 서술 금지.
  확실한 것: (i) A-DCVAE는 구조(non-zero along-fiber, 방향 정확도)를 복원, (ii) det-reg는 무정보 점으로
  붕괴. NPE 대비 우열은 v2로 판가름.
- 살 수 있는 것(v2 결과와 무관하게 유효): (i) **은닉 상태 궤적 I(t) 부산물**(NPE엔 없는 물리 해석성,
  Fig4), (ii) **3보장 설명**(B7), (iii) **ε_inc 자기일관 인증서**. decoupling이 진짜 이기는
  regime(외삽/강한 비식별/Sim2Real)은 **미실증(B10)** — 향후 과제.
- ⚠️ 별개 버그노트(이미 해결): 최초 50k 시도는 10k eval normalizer를 50k-학습 canonical 체크포인트에
  적용해 A-DCVAE를 부당하게 불리하게 함(mDI err 0.153). 전 방법을 동일 컨텍스트에서 학습하도록 수정.

### B5. 백본 불가지성 (why VAE 방어)  `[~]`  (2026-07-09 방향 확정)
- **핵심 판단: 3정리는 VAE에 대한 정리가 아니다.** 세 정리의 load-bearing 대상은
  ① condition-noise 증강, ② target-denoising 목적, ③ 추계적 alternating 샘플링 — 모두 **학습/추론
  ingredient**이지 생성 family가 아니다. 디코더 `g`를 "conditional denoiser"로 추상화하면 정리는
  백본 불가지가 된다. VAE는 이 가설을 만족하는 **최소 instantiation**일 뿐.
  - Thm2(target denoise=Tweedie score)는 사실상 **diffusion의 이론적 코어**와 동일 → 백본 바꿔도
    재사용(재설계 아님).
  - Thm1(condition-noise=Jacobian/Tikhonov)은 Bishop 1995류로 임의 mean-predictor에 성립. 단
    **condition-noise는 vanilla diffusion엔 자동이 아닌 추가 ingredient**(어느 백본에도 적용 가능).
  - Thm3은 **outer alternating chain**의 ergodicity → inner 조건부 sampler만 백본별로 바뀌고
    구조적 보장은 유지.
- **재작성 방침**: 정리를 디코더 `g`(denoiser) + 커널 수준에서 **한 단계 위로 재진술**, VAE ELBO는
  "그 `g`를 만드는 학습기"로 배치. → 정리를 버리지 않고 **일반화**(오히려 더 강한 기여).
- **프레이밍 역전**: 단순 백본은 약점이 아니라 **통제**다 — "최소 denoising instantiation(CVAE)만으로
  구조화된 posterior가 복원됨을 보여, 성공을 백본 용량이 아니라 3-ingredient 구조에 귀속." diffusion
  instantiation은 필수가 아니라 scaling 증거(옵션).
- 할 일(옵션): flow/소형 diffusion 디코더 1개 instantiation으로 "같은 3정리 성립"을 실증(B7과 묶음).

### B6. ε-비호환 gap 정량화 = **Theorem B**  `[x]`  (2026-07-10 RESOLVED — 증명 완료)
- ✅ **`theory_notes.tex` §11–12 완성**: `Lemma perturb`(W_2 고정점 섭동) + `Lemma opdisc`(연산자
  discrepancy ≤ L_P ε_H+ε_P) + `Theorem B`(steering: `W_2(ν*,p(θ|x_obs)) ≤ (L_P ε_H+ε_P)/(1-κ)`,
  Corollary B1: ε=0⟹정확) + `Prop inc`(ε_inc=0⟺compatible, 연산적) + `Remark nodouble`(보정3:
  ε_inc≤C·ε_app라 합산 금지) + `Remark edist`(σ/경계 분해). 정적검사 통과.
- ε_inc는 **정의 B(forward/backward sweep 불일치)** = `W_2(ν*⊗Q_H, μ*⊗Q_P)` — 학습모델만으로
  계산 가능한 **self-consistency 인증서**(B7 ablation에서 모니터). `Π←=π*`라 `ε_inc=W_2(Π→,π*)`.
- **3차 리뷰 반영(2026-07-10)**: (1.1) 최적커플링 **가측 선택** `Fact:msel`(Villani Cor 5.22) 신설,
  `Lemma:perturb`·`Lemma:opdisc`에 적용. (1.2) `Assumption:suptrue`(supp Π†⊆Ω) + 참조 조건부 고정
  version. (2.1) §9 TV 문장 정정(injective map 보존). (2.3) ε_dist 차원 정정(경계질량→RMS 사영변위).
  (2.2) ε_inc≤C·ε_app의 상수는 정직하게 후속 노트로 격하. `Lemma:sweep`에 K-kernel성 명시, Prop
  범주오류·a.e. 수정, `\vtheta`→`\bth`(컴파일 오류 해소). 정적검사 통과.
  ⚠️ 남은 cosmetic: §9–13은 plain θ,u / §1–8은 bold — 통일 pass 미실시(deferred).

- 문제: 두 조건부는 단일 결합분포에서 안 나옴(ε-incompatible). π*가 **무엇에** 수렴하며 참
  joint posterior와 얼마나 먼가?

- ⚠️ **논리적 함정 (반드시 회피)**: "denoising이 incompatibility를 극복 → π*=참분포"는 **과장**
  (옛 `L<1`과 동종). denoising은 **각 조건부를 개별적으로 정확하게** 만들 뿐, 두 조건부를 서로
  **호환(compatible)** 으로 만드는 마법이 아니다. incompatibility는 pair의 성질.

- **채택 프레임 (외부 피드백 2026-07-10 + 보정):** compatibility를 **가정이 아니라 측정 가능한
  결함량**으로. `W_2`에서 **joint** 대상 정리를 세우고 θ-marginal은 corollary.
  - 세 결함량: `ε_inc`(비정합; 학습 pair가 어떤 단일 joint에서도 안 나오는 정도),
    `ε_app`(참 조건부 근사오차), `ε_dist`(사영 경계질량 + σ-평활 왜곡).
  - **Perturbation 골격(Banach 고정점 안정성, W_2):** 연산자 조건별 `W_2`-수축계수 `κ<1`이면,
    고정점 `π*`(학습 `T`)와 `π_ref`(참조 `T_ref`)에 대해
    `W_2(π*,π_ref) ≤ (1/(1-κ)) sup_z W_2(T(z,·), T_ref(z,·))`.
    유도: `W_2(π*,π_ref)=W_2(Tπ*,T_refπ_ref)≤κW_2(π*,π_ref)+sup_z W_2(T,T_ref)`.
  - **θ-marginal 무료:** proj_Θ가 1-Lipschitz → `W_2(proj#π*, p(θ|x_obs)) ≤ W_2(π*, p_true)`.
  - **역할 분담:** Theorem A(TV, minorization) = **존재·유일(수축 불필요, σ>0만)**;
    W_2-수축(=옛 Thm1 Jacobian) = **오차 안정 상수 1/(1-κ)**. σ 긴장 해소(분업).
  - **denoising 역할:** `ε_app`(→ operator discrepancy)를 줄이는 학습 메커니즘. 공통 joint
    시뮬레이션이라 참 조건부는 호환 → `ε_inc→0` 극한에서 `π_ref=p_data`. (generic dependency
    network 대비 차별점.)

- ⚠️ **지도교수 보정 4곳 (외부 피드백에서 빠졌거나 위험한 부분):**
  1. **(가장 중요) 순수 W_2-수축은 Dirac로 붕괴 = 바로 그 mode collapse.** 결정론 수축맵의 고정점은
     점질량. 그러니 `κ<1`은 **조건부 평균의 Lipschitz성**에서 와야지 **노이즈를 줄여서** 오면 안 됨.
     `σ>0`은 `π*`의 **폭(비붕괴)**을 정하고 반드시 양수 유지. → **`κ<1`과 `σ>0`은 둘 다 필요, 서로
     다른 이유.** 이걸 안 박으면 "수축을 강화하라"가 곧 "붕괴하라"가 되는 자기모순. 논문 핵심과 직결.
  2. **존재-의존성 정직성:** W_2-수축(κ<1)을 가정하면 **Banach로 존재가 공짜**(P_2 완비) → 그 가정
     하에선 Theorem A(Doeblin)가 존재엔 불필요. ∴ **Theorem A의 고유가치 = 수축 가정 없이(가장 약한
     가정으로) 존재를 줌.** "A=존재, 수축=안정"을 쓰되 이 의존관계를 흐리지 말 것.
  3. **세 결함량 이중계상 주의:** 참 조건부는 호환이므로 `ε_inc ≤ ε_app`(참을 compatible Π로 택가능).
     → 셋을 단순 합하면 over-count. **2단 분해 권장:** `π* →(1/(1-κ))·ε_inc→ 최근접 compatible joint
     `Π_joint` →(app+dist)→ 참 posterior`. 즉 **`ε_inc`만 1/(1-κ)로 증폭**(고정점 이동), app/dist는
     별도(증폭 안 될 수 있음). 피드백의 `(1/(1-κ))(C1ε_inc+C2ε_app+C3ε_dist)`는 이 점 재검토 필요.
  4. **metric 전환 부기:** A는 TV, B는 W_2. `π*`는 동일 객체(유일 불변측도)라 무해하나, **수축·
     perturbation은 W_2에서만 성립**(TV에선 결정론맵이 수축 아님 — 두 Dirac은 항상 TV거리 1)임을
     명시. B가 W_2를 쓰는 **이유**가 이것.

- **held (a)/(b) 질문은 이 프레임에 흡수됨:** σ-평활 bias는 `ε_dist`로, 참조 연산자는 "최근접
  compatible Π"로 → (a)무노이즈 vs (b)σ-평활 참 조건부의 이분법이 3결함 분해로 대체.

- 할 일(순서): (1) ✅ **W_2-수축 보조정리 완료**(`theory_notes.tex` §9–10: Wasserstein 준비 +
  `Lemma condlip`(조건부 W_2-Lipschitz, N4 상쇄) + `Lemma sweep`(κ=L_HL_P 수축, Banach로 ν* +
  rate) + `Remark noises`(붕괴 역설 소멸) + `Remark depend`(존재-의존성 정직성)). 정적검사 통과.
  (2) `ε_inc` 정의 확정 (inf-over-Π vs forward/backward sweep 불일치). ← **다음 결정**
  (3) 2단 오차전파(perturbation) 부등식 조립 = Theorem B 본체.

### B7. 3정리를 격리하는 ablation (이론의 실증)  `[x]`  (2026-07-14 RESOLVED — 10k + 50k 캐노니컬 모두 실행 완료)
- ✅ 구현·실행: `prob_models/paper/experiments/b7_ablation.py`(5 seed × 3 variant, 데이터/split
  고정·학습randomness만 변동). `figure_b7_ablation.pdf`, `results/paper2_experiments/b7/b7_metrics.json`.
  **10k-scale(exploratory)에 이어 50k/10000-epoch-cap/patience=200 캐노니컬 스케일**(사용자의 기존
  `iter_cvae` 체크포인트와 동일 설정)로 재현 — 15/15 run 정상 종료(~17.2h, 무인 백그라운드 실행).
- **결과 (mean±std over 5 seeds, 10k → 50k canonical):**
  - (a) **N1 condition on/off → Thm1 안정성 ✅**: κ full **0.39→0.41** → no_cond **0.62→0.71**(κ=1
    한계로 접근하는 폭이 캐노니컬에서 더 뚜렷함). divergence_rate=0 양쪽 스케일 모두(경계사영으로 발산
    아님) → 효과는 **rate κ**이지 blow-up 아님(B12 정정과 정합).
  - (c) **N4 추계 vs 결정론 핑퐁 → Thm3 붕괴우회 ✅ (가장 견고, 스케일 불문 재현)**: full/no_target에서
    결정론 핑퐁은 along-fiber std를 **0.000**으로 붕괴, 추계는 **~0.7-0.9** 유지.
    **캐노니컬에서 새로 드러난 뉘앙스**: no_condition의 결정론 핑퐁은 붕괴가 불완전(det std
    **0.49→1.72**) — κ가 1에 가까울수록(약한 수축) 결정론 사상 자체도 고정점 수렴이 느려짐을 보여줌.
    N1/κ(수축 rate)와 N4/injection noise(비붕괴 폭)가 **서로 다른 역할**이라는 §E 논증을 강화하는
    증거로 활용 가능.
  - (b) **N2 target on/off → Thm2 방향 ❓(격리효과 미미, 양쪽 스케일 재현)**: no_target의 mDI
    err(10k:0.04→50k:0.02)·spread가 full과 유사 → **타깃노이즈의 고립된 이득 관측 안 됨**. 정직히
    보고(B12: Thm2=배경원리라는 자체 진단과 정합, load-bearing 아님).
  - **ε_inc = W₂(sweep(π*), π*) ≈ 0.03** 전 variant·양쪽 스케일 → π*가 near self-consistent.
- ⚠️ 선결 C1 버그: 2026-07-13 수정 완료(`CONDITION_NOISE_STD_*`로 통일 + C4 추론노이즈 배선).
- 실행 노트: 캐노니컬 스케일 재현 중 `setsid`+heredoc 조합으로 백그라운드 실행 시 원인불명 조기종료
  발견(epoch 20) — 이 세션에서 이미 검증된 `nohup python -m <module> ... &`(파일 기반 스크립트,
  setsid/heredoc 없이) 방식으로 전환해 해결. 향후 장시간 백그라운드 실행 시 이 패턴 사용 권장.

### B8. 현대 이웃 문헌 포지셔닝 (SBI + score-based MCMC)  `[x]`  (2026-07-14 Related Work 작성)
> **2026-07-14 반영**: `main.tex` `\subsection{Related Work}` 4문단 — (1) SBI/NPE(cranmer, papamakarios2016fast,
> greenberg2019automatic, lueckmann2021benchmarking, papamakarios2021normalizing), (2) score-SDE/diffusion
> (song2019/2021, ho2020, lipman2023flow), (3) score posterior sampling+split-Gibbs(chung2023, kadkhodaie2021,
> vono2019, coeurdoux2024) = **가장 가까운 현대 이웃**, (4) compatible conditionals/dependency networks
> (heckerman2000, arnold2001). 차별점: decoupling+은닉상태 물리구조+비식별 전용+계산가능 ε_inc 인증서.
> ⚠️ bib 12개 신규 = 지식 기반 작성, **서지정보(저자/venue/권/페이지) 컴파일 전 검증 필요**.
- 문제: 고전 인용(Bishop95/Vincent11/Heckerman00)은 문제 아님. **진짜 리스크는 "현대 이웃"을
  안 걸치는 것** — 잘 아는 리뷰어는 우리가 최신 도구를 몰라서 조각조각 쓴다고 공격한다.
- 반드시 걸쳐야 할 현대 이웃:
  1. **Score-based / SDE 생성모델**: Song&Ermon(2019), Song et al.(2021 SDE), Ho et al.(DDPM) —
     "denoiser=score"의 현대적 본거지. Thm2가 여기 배경임을 인정하고 인용.
  2. **Score/diffusion 기반 posterior 샘플링 & MCMC**: plug-and-play priors, DPS/ΠGDM,
     **split Gibbs sampler**(Vono et al.; Coeurdoux) — "score로 posterior를 교대 샘플"하는 우리와
     **가장 가까운 현대 계열**. 이걸 안 걸치면 치명적.
  3. **SBI**: Papamakarios&Murray(SNPE), Greenberg(APT), Cranmer review, Lueckmann.
  4. **Flow matching / stochastic interpolants**: 생성 프런티어(=why VAE 방어와 연결).
  5. **학습된 조건부의 비호환/편향**: dependency networks(Heckerman) + compatible conditional
     specification(Arnold–Castillo–Sarabia). → Thm3b(incompatibility gap)의 계보.
- 할 일: 위를 Related Work로 명시하고, 우리 차별점(**decoupling + 은닉상태 물리구조 + 비식별
  전용**)을 대비.

### B12. 이론 altitude 재조정 (theorem 격상 과잉 여부)  `[~]`  (2026-07-09 판단)
- **진단**: 세 "정리" 중 둘은 **오늘날 routine한 사실을 정리로 과격상**한 것. weight-decay를 정리로
  안 쓰는 것과 같은 문제.
  - **Prop1(condition noise=Tikhonov)**: Bishop 1995. 완전 textbook. **정리 아님 → 배경/설계선택으로
    강등.** 게다가 체인 boundedness의 실제 보증은 **경계 사영(compact Ω)** 이지 Jacobian 페널티가
    아님(부분 redundant). 강등 근거 확실.
  - **Thm2(denoiser=score)**: Vincent2011/Tweedie. 현대 diffusion의 배경 상식. **정리 아님 →
    leverage하는 알려진 원리로 배경화**, 인용 강화.
  - **Thm3**: routine 부분(3a 기하 ergodicity)은 강등, **비routine 부분(3b: 비호환 조건부에서 π*의
    편향)은 오히려 저개발** → 여기에 이론 무게를 옮김.
- **진짜 theorem감 (비routine 질문)**:
  - Q-A: 교대 추계 샘플러가 붕괴 대신 **올바른 support(비식별 fiber)** 를 복원하는가, 참 posterior
    대비 편향은?
  - Q-B: **두 학습 조건부의 비호환성이 π*를 참 결합사후에서 얼마나 밀어내는가 — 상한**(pseudo-Gibbs
    편향의 정량화; 실제로 저개발된 문제). ← 가장 매력적.
  - Q-C: 노이즈 스케일 `(σ_H,σ_P)`가 만드는 **bias–variance/mixing trade-off** 특성화.
- **권고(venue-contingent)**: AISTATS면 Q-B(또는 Q-A) **하나를 진짜 정리**로 완성 + 나머지는 배경.
  TMLR이면 "방법+정직한 분석+강한 실험(해석적 참조 posterior 대비)"로 **headline 정리 없이도** 충분.
  어느 쪽이든 **routine 2개(Prop1/Thm2)는 배경으로 강등**.

- **[정정 2026-07-09] "경계 사영이 수렴의 진짜 이유"는 틀렸음(점 dynamics ≠ 측도 dynamics).**
  - 점 dynamics(무노이즈 평균맵 `m`): compact ⇏ 수렴. compact 위 chaos(stretch-and-fold) 가능.
    수렴엔 수축(Jacobian/Lipschitz<1) 필요 — **student 지적이 옳음**.
  - 측도 dynamics(노이즈 마르코프 체인 `μ_{k+1}=T μ_k`): **compact Ω + 양의 가우시안 커널**이면
    커널이 `k(x,y)≥δ>0`(compact 위 연속양수는 양의 최소값) → **Doeblin minorization → uniform
    ergodicity**, 유일 `π*` + TV 수렴 `(1-δ)^k`. **평균맵이 chaotic이어도 분포는 수렴**. 노이즈가
    점 흐름은 못 잡아도 측도 흐름은 정규화함.
  - ∴ **Jacobian 제어(Prop1)의 진짜 역할 = 존재/유일성이 아니라 (i) tight한 Wasserstein 수축
    RATE `L^k`(synchronous coupling), (ii) 저노이즈 영역 강건성(δ→0 as σ→0이라 Doeblin 붕괴 →
    chaos 극한 복귀), (iii) 덜 퍼진 π* 집중.** 발산 방지가 아님.
  - 정정: (a) "경계 사영이 수렴의 이유"는 **boundedness/존재의 이유**일 뿐 rate/quality 아님.
    (b) "Jacobian 페널티가 사영과 redundant"는 **철회** — 사영=boundedness/존재, Jacobian=rate/저노이즈.
  - **Thm 3a 재설계**: **두 경로**로 진술 — ① compact+양의커널 ⟹ uniform ergodicity(chaos에 강건,
    존재/유일, rate 약함); ② 평균맵 수축(Jacobian 제어) ⟹ Wasserstein 기하수축(tight rate). 이게
    더 정교·정직. condition-noise 역할을 "발산 방지"가 아니라 "**mixing rate/저노이즈 강건성**"으로
    서술. 노이즈가 마르코프 체인을 정규화한다는 현대 계열(Langevin/annealing, σ→0 mixing 붕괴)과 연결.

### B9. 이론 엄밀화  `[ ]`
- Thm1: dissipativity를 구체 Lyapunov 함수로. Thm3: 2-Wasserstein 수축상수·조건 명시.
- 목표 수준을 정해야: "직관적 정당화(현재)" vs "정리(定理)로서의 증명". Venue가 이를 좌우.

### B10. Sim2Real 주장 정리  `[ ]`
- abstract/conclusion이 Sim2Real을 약속하나 실험 없음. **실험을 하든가, 주장을 내리든가.**

### B11. 실패모드/한계 정직성  `[ ]`
- 언제 무너지는가(강한 practical non-identifiability, 다봉 3개 이상, OOD 노이즈=현재 Fig5 약점).
- Fig5의 "high sensitivity"를 한계로 정직히 프레이밍(이미 부분적으로 함).

---

## C'. 진행 현황 (2026-07-13)
- ✅ **이론 완성**: `theory_notes.tex`에 Theorem A(존재/ergodicity)·sweep 수축(rate)·Theorem B
  (steering+ε_inc), 3차 리뷰까지 반영. (남은: ε_inc 상수 증명, plain/bold 표기 — companion 작업 시.)
- ✅ **Task 2 계획서**: `THEORY_PAPER_PLAN.md` (JMLR 1순위, 3-공백, 구조, 분업).
- ✅ **Task 1(이론 이식)**: `04_method.tex` §3.4 축약 A/B + `appendix.tex` 증명스케치, companion
  (`adcvae-theory`) 인용. bib에 adcvae-theory·bishop1995training 추가.
- ✅ **Track W 1차(2026-07-14)**: intro/abstract/background/contributions 재작성(B1), Related Work
  신규(B8), W4 정직성(PICP↔MPIW 방향성 + Limitations/Sim2Real 완화 + B11 실패모드), Fig1 캡션·설계스펙.
  정적검사 통과(미정의 `\xobs` 수정 포함).
- ⏳ **미완(Track E 의존)**: Fig1(실제 PDF)/Fig2/4/5, B7 ablation의 ε_inc 수치, 참조 posterior 대비
  along/across-fiber 정량비교, 강baseline(B4: NPE/flow + 결정론 iterative) — experiment1.tex에 Track E
  placeholder 주석으로 표시함.

## C. 다음 액션 후보 (우선순위)
1. **B1+B4+B8 (프레이밍 재설계)** — 논문 정체성을 바꾸는 최상위 결정. 여기부터 논의.
2. **B2+B3 (검증 인프라)** — toy 시스템 + reference posterior. 모든 정량 주장의 토대.
3. **B7 (이론-실증 ablation)** — 기여의 핵심 증거.
4. 그 다음 writing(W4 포함)·figure.

## E. 개념 노트 — 노이즈 4종 분류 (σ 혼동 방지)  (2026-07-10)

코드(`trainer.py`, `infer.py`, `models.py`)에는 서로 다른 **네 개**의 확률원이 있다. 정리에서
"σ/noise"라 할 때 반드시 어느 것인지 명시한다.

| # | 이름 | 코드 위치 / config | 무엇에 주입 | 정리 역할 |
|---|---|---|---|---|
| N1 | **Condition noise** `σ_c` | `CONDITION_NOISE_STD_{Y,P}` (train); ⚠️C1버그로 현재 0.05 고정 | 디코더의 **조건 변수**(H-net엔 θ, P-net엔 x_hid) | **κ (W_2 수축)** — 조건-Jacobian 정규화. **Theorem 1→B의 수축 엔진.** |
| N2 | **Target noise** `σ_t` | `TARGET_NOISE_STD_{Y,P}` (train) | **인코더에 들어가는 타깃**(clean 복원) | denoising=score(Theorem 2). 학습되는 조건부의 평활/정확도 → `ε_app` |
| N3 | **CVAE latent** `z` | 학습: reparam `μ+εσ`; 추론: `z~N(0,I)` prior | 디코더 latent 입력 | 조건부의 **다봉 표현력**(mixture) |
| N4 | **Injection noise** `σ_infer` | `INFER_NOISE_{Y,P}` (inference) | 디코더 **출력**(`ŷ=g+σ_infer ξ`) | **Theorem A minorization**(full-support 커널) + `π*`의 **폭(비붕괴)** |

**핵심 매핑 (보정 1의 정밀화):**
- **수축 `κ<1` ← N1 (Condition noise, 학습).** `σ_c↑ → ∂g/∂(조건) Jacobian↓ → κ↓`. `κ=κ_Hκ_P`
  (합성). Theorem A의 minorization과 **무관**.
- **비붕괴 폭 ← N4 (Injection, 추론) + N3 (latent).** `σ_infer>0`이 `π*`를 점질량이 아니게 유지.
  `σ_infer↓ → π*` 붕괴(=mode collapse).
- ∴ **"수축 강화(σ_c↑)"와 "붕괴 방지(σ_infer>0)"는 서로 다른 knob → 충돌 없음.** 보정 1의 역설이
  노이즈를 분리하는 순간 **소멸**한다. (전에 "σ" 하나로 뭉쳐서 역설처럼 보였을 뿐.)
- **Theorem A의 σ = N4** (커널 밀도 하한은 `N(u;g,σ_infer²I)`의 full support에서 옴; N3만으론
  이미지 저차원 매니폴드라 부족).
- **ε_dist 평활 floor ≈ O(σ_infer)+O(σ_t) + π*(∂Ω)** — N4는 0으로 못 보냄(A·비붕괴가 요구) →
  **소멸 불가**하지만, 비식별 ridge 폭에 비하면 작음(우리 관심 영역에서 무해).

## D. 개념 노트 — Doeblin minorization (직관)

**한 줄**: "매 스텝, 어디서 출발했든 일정 확률 `ε`로 **같은 분포 `ν`에서 재출발(restart)**한다"면,
초기조건 기억이 기하적으로 지워져 유일 정적분포로 `(1-ε)^k` 수렴한다.

- **조건(minorization)**: 모든 `x`에 대해 `P(x,\cdot) \ge \epsilon\, \nu(\cdot)` (공통 겹침 `ε·ν`).
- **왜 우리에게 성립**: compact `Ω` 위 가우시안 커널 `k(x,y)=\mathcal{N}(y;m(x),\sigma^2 I)`는
  `Ω\times Ω`에서 연속·양수 → **양의 최소값 `δ>0`**. ∴ `P(x,A) \ge \delta\,\mathrm{Leb}(A)`,
  즉 `\nu=`정규화 Lebesgue, `\epsilon=\delta|\Omega|`로 minorization 성립. 경계사영은 경계에
  atom만 더할 뿐 내부 하한 안 깸.
- **의미**: 노이즈가 "모든 점에 공통된 재출발 확률"을 심어 초기조건 기억을 지움. **수축(L<1) 없이도**
  ergodicity. `σ→0`이면 `δ\to 0` → Doeblin 붕괴(저노이즈 chaos 극한). Meyn–Tweedie 특수형.
- **철학 부합**: 가정(compact+양의커널)이 알고리즘 구성으로 **보장됨** → "가정 말고 증명" 실현.

### 로그
- 2026-07-09 — 최초 작성. 결정론 논문 상태(SIAM/robustness-thesis/미완성/θ* non-attracting) 확인
  후 5개 질문에 대한 전략 판단 + B1–B11 공백 목록 정리.
- 2026-07-09 — Ha J 증명으로 B2 완결/B3 near-solved, scope 확정(일반방법·OGTT=예시), 이론 altitude
  재조정(B12), 점/측도 dynamics 정정, 이론 뼈대 결정화(A''): "수렴 증명(Thm A, Doeblin) + 방향 조종
  (Thm B, denoising)"의 2단 구조 = 전작(spectral norm + teacher forcing)의 확률론적 평행. Doeblin
  개념노트(§D) 추가.
- 2026-07-09 — `theory_notes.tex` Rev.3: 2차 리뷰 반영. Fact(kernel 측정가능성)·Lemma(averaging
  bound) 신설, Fact(Leb(∂K)=0) self-contained dilation 증명, Fact(TV 완비성) Radon–Nikodym+Riesz–
  Fischer로 교체, Ionescu–Tulcea 임의 가측공간으로 완화, coupling의 "independently"→조건부 귀납,
  Fact(projection) 미시계산 명시, Step3 nonneg-measure 도미네이션 명시, semigroup 항등식 명시,
  인용 판본-의존 번호 제거(4개 참고문헌, 전부 인용·자기완결 증명). 정적검사 통과.
