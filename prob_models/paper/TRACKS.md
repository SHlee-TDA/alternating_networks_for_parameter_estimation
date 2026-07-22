# TRACKS — 병렬 워크스트림 & 세션 kickoff

> `prob_models`(Paper 2, A-DCVAE) 프로젝트의 남은 일을 **3개 병렬 트랙**으로 나누고, 각 트랙을
> 새 세션(콜드 스타트)에서 시작할 수 있게 todo·주의사항·kickoff 프롬프트를 정리한다.
> 최초 작성 2026-07-13. 참조: `PAPER_BACKBONE.md`, `DISCUSSION.md`, `THEORY_PAPER_PLAN.md`,
> `theory_notes.tex`.

---

## 0. 모든 세션 공통 주의사항 (콜드 스타트 필수)

1. **스코프**: 이 작업은 `prob_models` = **Paper 2 (A-DCVAE, 확률론적)** 이다. 저장소 root의
   `CLAUDE.md`는 **결정론적(spectral normalization) 논문**용이고 거기서 "prob_models는 out of
   scope / estimator는 deterministic" 이라 한 것은 **그 논문 얘기**다. **이 작업엔 적용되지 않는다.**
   반드시 먼저 `prob_models/paper/PAPER_BACKBONE.md`를 읽어 스코프를 확정할 것.
2. **파일 위치**: 논문·이론 소스는 **메인 작업 트리**
   `/home/shlee/projects/alternating_networks_for_parameter_estimation/prob_models/paper/`에 있다
   (대부분 git 미추적/미커밋). `.claude/worktrees/*`에는 없을 수 있음.
3. **먼저 읽을 문서 (순서대로)**:
   - `PAPER_BACKBONE.md` — 논문의 주장·방법·기여·그림현황 (중심 뼈대).
   - `DISCUSSION.md` — 공백 목록 B1–B12 + 모든 설계 결정 + **§E 노이즈 4종 taxonomy** + §D Doeblin.
   - `THEORY_PAPER_PLAN.md` — companion(JMLR) 이론논문 계획.
   - `theory_notes.tex` — 완전 형식화된 이론(Thm A/B, 3차 리뷰 반영). backstage 정본.
4. **LaTeX 툴체인 없음**(세션 내). 컴파일 대신 **정적 검사**: `\begin/\end` 환경 짝, 인용키 존재,
   `\ref/\label` 정합, 미정의 매크로. (사용자가 로컬에서 컴파일함.)
5. **작업 rhythm**: 이 프로젝트는 "논의 → 합의 → 작성" 순으로 진행해 왔고, 수학은 매우 엄격히
   리뷰된다. 큰 변경 전 설계를 먼저 제시하고, 정적 검사로 무결성을 확인할 것.
6. **핵심 이론 요약(콜드 스타트용)**: 존재는 **Theorem A**(노이즈 minorization→Doeblin→uniform
   ergodicity, TV, 가정 없이), rate는 **sweep 수축**(condition-Lipschitz→W_2, κ=L_HL_P), 정확성은
   **Theorem B**(고정점 perturbation: `W_2(ν*,p(θ|x_obs))≤(L_Pε_H+ε_P)/(1−κ)`) + 계산 가능한
   self-consistency 인증서 `ε_inc=W_2(Π→,π*)`.

---

## 1. 현재 상태 (완료된 것)
- ✅ 이론 완성(`theory_notes.tex`): Thm A / sweep 수축 / Thm B + ε_inc, 3차 정밀 리뷰 반영.
- ✅ 논문 이론 이식(축약): `sections/04_method.tex` §3.4 (Thm A/B 축약+직관), `sections/appendix.tex`
  (증명 스케치), companion `\citep{adcvae-theory}` 인용. bib에 adcvae-theory·bishop1995training.
- ✅ 표기/LaTeX 정리, 비활성 초안 삭제, Fig3 PDF·Table1 값 존재.
- ✅ B2(비식별 `S_I·σ` fiber) 증명 확보(Ha et al. JCEM 2025, bib: `ha2025disposition`).
- 커밋: `main` `f1eb185`에 초기 논문 소스. 이후 변경은 미커밋(사용자 확인 후 커밋).

---

## 2. Track E — 실험 (본 논문 figure + companion 수치예시)

**목표**: 본 논문의 빈 figure(Fig 1/2/4/5)와 정량 근거를 채우고, 이론(A-contraction·Thm B)을
경험적으로 실증. E-b/c는 companion 논문(T2-3)에서 재사용.

> **진행 (2026-07-13, 실험 세션):** 구현 스크립트는 `prob_models/paper/experiments/`에
> (`_context.py` 비대화형 로더, `_metrics.py` sliced-W2/coverage, `b3_reference_posterior.py`,
> `b7_ablation.py`+`b7_replot.py`, `b4_baselines.py`, `phase4_figures.py`). 산출물: metric JSON은
> `results/paper2_experiments/{b3,b7,b4,fig}/`, figure PDF는 `prob_models/paper/figures/`.
> 데이터/체크포인트는 메인 트리 심링크 재사용(50k sims + 기존 iter/single/det 체크포인트).

**Todo**
- [x] **E0 (선결): C1 버그 수정** — 완료(2026-07-13). `trainer.py`를 config 정본 키
  `CONDITION_NOISE_STD_{Y,P}`로 통일. 겸사겸사 C4(추론 N4)도 배선: `main.py`가 `INFER_NOISE_{Y,P}`를
  모델 속성으로 주입 → 추론 주입노이즈가 config대로 작동(추계↔결정론 ablation 선결). smoke로 파이프라인 검증.
- [x] **E1 (B3): OGTT 해석적 참조 posterior** — 완료. `(S_I,σ)` 격자 × BDF forward-sim × 가우시안
  우도(mDI=S_I·σ) × lognorm prior → 2-D 참조분포. **8관측 평균: MAP mDI 상대오차 2.4%±2.0%,
  across-fiber CV 10.0%±1.4%** (B2 비식별 증명의 수치 실증). π* 비교: sliced-W2 0.20±0.14,
  ref-HPD95 coverage 18.8%±2.6%, W1(logmDI) 0.489±0.011 — π*가 fiber 방향은 따르나 과대분산
  (mDI CV 80% vs 10%)+일관된 offset(정직한 근사갭). `figure_b3_reference_posterior.pdf`.
- [x] **E2 (B4): 강baseline** — **완료(2026-07-15), v2로 최종 확정**. self-contained conditional
  RealNVP NPE 단일망 + 결정론 MSE regressor(collapse: along-fiber std 0). **결정론 iterative는
  iter_det 체크포인트 대신 깨끗이 수렴하는 MSE regressor 사용** — G0 병리 전시 회피(§주의).
  - 10k-scale: decoupling이 단일망 NPE를 명확히 이기지 못함(sliced-W2 무승부, coverage는 NPE 우위).
  - 50k 캐노니컬 v1: 순위 뒤집힘(A-DCVAE 우위)이었으나 patience 불균형(flow/reg=40, A-DCVAE=200)
    confound 있었음.
  - ✅ **v2(patience=200 전체 통일) 완료 — confound 기각, 결론 확정**: patience를 늘렸더니 NPE-flow는
    **오히려 더 나빠짐**(coverage 0.610→0.572, along-fiber spread 1.995→2.105, sliced-W2
    0.481→0.533) — "flow가 덜 학습돼서 불리했다"는 가설이 틀렸음을 직접 확인. A-DCVAE는 v1→v2 사실상
    동일(0.170/0.712/0.016, 재현성 확인). **최종 결론: 캐노니컬 스케일에서 A-DCVAE가 sliced-W2(0.170
    vs 0.533)·방향(0.016 vs 0.013 근접)에서 우위, NPE의 높은 raw coverage(0.57)는 과대분산의 신호일
    뿐 정확도 우위가 아님.** 10k-scale에서는 반대였다는 점도 함께 명시(스케일 의존성).
  - 논문 기여 프레이밍 갱신: "decoupling은 metric 대신 해석성만 판다"가 아니라 **"매칭된 학습예산 하
    캐노니컬 스케일에서 decoupling이 구조 복원 정확도(sliced-W2·방향)에서도 우위"**로 격상 가능(단
    스케일 의존성은 정직하게 명시). 은닉상태 해석성(I(t))·3보장 설명·ε_inc 인증서는 보완 논거로 유지.
  - `figure_b4_baselines.pdf` (v2 최종 반영 완료). 정규화기 불일치 버그는 별도로 이미 수정됨(모든
    방법을 동일 컨텍스트에서 학습).
- [x] **E3 (B7): 3-보장 격리 ablation** — 완료(5 seed × 3 variant, **10k + 50k 캐노니컬 스케일 모두
  실행, 15/15 run 정상 종료**, ~17.2h). 이론과 정합, 캐노니컬 스케일에서도 재현:
  (a) N1 condition off → **κ 0.41→0.71↑**(50k, Thm1 수축 저하 신호가 10k보다 더 뚜렷함, div=0: 발산
  아닌 rate 효과), (c) 추계 vs 결정론 핑퐁 → **along-fiber std 0.72 vs 0.000**(full/no_target, Thm3
  붕괴 우회, 견고). **50k에서 새로 드러난 점**: no_condition의 결정론 핑퐁도 붕괴가 불완전(det std
  0.49→1.72) — κ가 1에 가까울수록 결정론 사상 자체도 고정점에 덜 수렴함을 보여줌(N1/κ와 N4/비붕괴가
  별개 역할이라는 논증 강화). ε_inc≈0.03(자기일관, 양쪽 스케일 동일). ⚠️ (b) N2 target off은 여전히
  **격리 효과 미미**(mDI err ≈ full) — 정직히 보고(B12: Thm2는 배경원리). `figure_b7_ablation.pdf`.
- [x] **E4: figure 생성** — 완료. Fig2(MCMC trace+ACF, lag-1 autocorr 0.012 빠른 mixing),
  Fig4(predictive: fiber 샘플 6개가 같은 G 재구성 / hidden I 발산 320→45, 비식별 실증),
  Fig5(noise sensitivity: dual std 0.33→0.64 상승 vs single flat 0.003). `phase4_figures.py`.

**주의사항**
- 환경: `conda activate vision_task`. 학습은 `config.py` 설정 후 `python prob_models/main.py`
  (`--system ogtt_simul --run_baseline true/false`). 체크포인트 필요.
- **결정론 baseline은 "개념적 collapse(평균이 무정보)"를 보여야** 함 — 결정론 논문의 미해결
  최적화 병리(G0의 θ* non-attracting)를 전시하지 말 것. 잘 수렴시키거나 conditional-mean oracle로
  대체해 "잘 수렴해도 평균이라 무정보"를 보이는 게 강함.
- `models.py`의 말단 `Tanh`(param)/`Sigmoid`(hidden)가 정규화 파라미터/상태를 절단할 위험(C2/C3) —
  정규화기와 정합 확인.
- 재현성: seed ≥5, mean±std, 파라미터 오차는 물리 단위. metric JSON, figure PDF.
- `mDI=S_I·σ` fiber는 증명됨(cite `ha2025disposition`); `figure3.py`의 fiber는 정확.

**핵심 파일**: `prob_models/{main,trainer,infer,analysis,metrics,models,config}.py`, `master_train.py`,
`figure3.py`, `systems/ogtt_simul.py`.

---

## 3. Track T — Companion 이론 논문 (JMLR)

**목표**: `theory_notes.tex`를 독립 JMLR 논문으로. 계획: `THEORY_PAPER_PLAN.md`.

**Todo (THEORY_PAPER_PLAN.md §9)**
- [x] **T2-1: `ε_inc ≤ C(ε_H+ε_P)` 상수 실제 증명** (was deferred in `Remark:nodouble`). 참 조건부
  W₂-Lipschitz(Ass:truereg) 하에서 `C=C(κ,L_P,L_H†,L_P†)` 명시 → `theory_notes.tex`에 Thm:incbound으로
  추가(+Lem:assembly, Rem:incconst). 정적 검사 통과. [2026-07-14]
- [ ] **T2-2: approximate-MCMC 포지셔닝 절** — Mitrophanov(2005), Rudolf–Schweizer(2018),
  Johndrow–Mattingly, Medina-Aguayo et al. 대비. 차별점: 학습 dual conditional + `ε_inc` + 노이즈
  taxonomy. bib 확충. **(안 걸치면 치명적.)**
- [ ] **T2-3: 수치 예시** — E-a(선형가우시안 compatible→Cor B1·rate), E-b(통제된 비호환→Thm B
  bound·ε_inc 추적), E-c(비식별 toy→붕괴없는 support 복원). Track E의 B7과 공유.
- [ ] **T2-4: JMLR 논문 초안 조립** — `theory_notes.tex` → 논문 구조(계획 §5), **표기 통일**
  (§9–13 plain θ,u → §1–8 bold와 일치).
- [ ] **T2-5: 제목 확정**.

**주의사항**
- `theory_notes.tex`는 3차 정밀 리뷰를 통과한 정본 — 수정 시 같은 엄밀도 유지(`Fact:msel` 가측
  선택, `Assumption:suptrue` 등). LaTeX 툴체인 없음 → 정적 검사.
- **일반성 유지**: OGTT-특화 서술 최소화(응용 논문과 중복/self-plagiarism 회피). OGTT는 motivating
  example로만.
- 개별 도구는 표준 → 신규성은 (대상·`ε_inc`·통합 프레임)에. 과대주장 금지(전작들의 `L<1` 교훈).

**핵심 파일**: `theory_notes.tex`, `THEORY_PAPER_PLAN.md`.

---

## 4. Track W — 본 논문 writing/서사

**목표**: 본 논문(ICLR/AISTATS/TMLR) 서사·related work·정직성 보강.

**Todo**
- [x] **B1: intro 서사** — "공유 연산자 T, 무엇에 수렴하는지 고친다": 존재(Thm A)·정확성(Thm B)를
  전작(spectral norm + teacher forcing)과 평행하게. 결정론 방법은 baseline+브릿지(companion 인용,
  load-bearing 아님). regression-to-mean은 '끝'이 아니라 '시작(inciting incident)'으로.
  **(2026-07-14 완료)** abstract·intro ¶2–¶5·contributions 재작성, mixture identity \eqref{eq:mixture},
  background "our prior work"→"deterministic decoupled baseline"(고정점=E[θ|x_obs] inline, load-bearing X).
- [x] **B8: related work** — SBI/NPE(Papamakarios, Cranmer), split Gibbs/score-MCMC(plug-and-play,
  DPS, Vono), dependency networks(Heckerman), flow matching. 차별점 명시.
  **(2026-07-14 완료)** main.tex `\subsection{Related Work}` 4문단 + bib 12개 신규(⚠️서지 검증 필요).
- [x] **W4: 정직성 보강** — Table 1의 PICP↔MPIW 트레이드오프("과대 불확실성" 반론) 서사;
  B10(Sim2Real 주장: 실험 없으면 완화); B11(실패모드/한계).
  **(2026-07-14 완료)** experiment1.tex "Coverage vs sharpness" 문단(방향성 검증=따라-fiber, 참조
  posterior는 Track E placeholder); conclusion에 Limitations 문단(B10 Sim2Real 완화, B11 실패모드).
- [x] **Fig1 teaser 설계** — single net blob vs Dual CVAE crescent.
  **(2026-07-14 완료)** 캡션 재작성(blob→conditional-mean 붕괴점 vs crescent fiber) + LaTeX 설계
  스펙 주석(2패널, 공유 S_I–σ 평면, 별=참값, fiber 곡선). \fbox placeholder 유지(실제 PDF는 추후).
- [x] **정적검사 부수 수정** — 미정의 매크로 `\xobs`(theory/appendix 4곳 사용, 정의 없음=컴파일
  오류)를 main.tex preamble에 `\newcommand{\xobs}{\mX_{\mathrm{obs}}}`로 정의. cite키·ref/label·
  env짝 전부 통과.

**주의사항**
- 스코프(A''): **일반 비식별 역문제 방법, OGTT는 한 예시**. `mDI`는 인용과 함께 부수적으로만
  (ML 독자에게 과도하게 specific해 보이지 않게). Venue: AISTATS/TMLR(현실), ICLR/NeurIPS는 baseline·
  novelty 보강 후.
- 이론은 축약본만(companion `\citep{adcvae-theory}` 인용). 과대주장 금지.
- **Track E의 결과가 있어야 experiment 서술·Table·Fig 최종 확정** → W는 E와 부분 의존.

**핵심 파일**: `iclr2026/main.tex`, `sections/{01_introduction 없음→main.tex inline, prob_formulation,
04_method, experiment1, appendix}.tex`.

---

## 5. 트랙 간 의존성 (완전 독립 아님)
- **E → W**: 실험 결과·figure·`ε_inc` 수치가 있어야 W의 experiment 서술·Table·Fig 확정.
- **E ↔ T**: B7 ablation(E3) ≈ companion 수치예시 E-b/c → **공유**. 한쪽 결과를 다른 쪽이 재사용.
- **T → W(그리고 본 논문 전체)**: companion이 진행돼야 본 논문의 "축약+인용" 전략이 성립
  (이미 `adcvae-theory`로 인용 중이므로 방향은 고정).
- 조율점: 세 트랙 모두 이 `TRACKS.md`·`DISCUSSION.md`를 갱신하며 진행(로그 남기기).

---

## 6. Kickoff 프롬프트 (새 세션에 복사)

각 프롬프트는 콜드 스타트 자기완결형. §0 공통 주의사항을 먼저 읽게 한다.

### ▶ Track E (실험) 프롬프트
```
prob_models(Paper 2, A-DCVAE 확률론적 파라미터 추정) 프로젝트의 "실험 트랙"을 진행합니다.
먼저 prob_models/paper/TRACKS.md의 §0(공통 주의사항)과 §2(Track E)를 읽고, PAPER_BACKBONE.md·
DISCUSSION.md(특히 §E 노이즈 taxonomy, B3/B4/B7)를 참고하세요.

중요: 저장소 root의 CLAUDE.md는 결정론 논문용이라 "prob_models out of scope"라 하는데, 이 세션은
바로 그 prob_models(Paper 2)를 작업하므로 그 스코프 제한은 적용되지 않습니다.

목표: 본 논문의 빈 figure와 이론 실증. 순서:
(1) C1 버그 수정(trainer의 COND_NOISE_STD_* ↔ config의 CONDITION_NOISE_STD_* 키 통일).
(2) B3 OGTT 해석적 참조 posterior 스크립트(p(S_I,σ|G)∝L(S_I·σ)·prior, 격자+forward-sim).
(3) B7 3-보장 격리 ablation(condition/target noise on-off, 추계 vs 결정론), ε_inc=W_2(Π→,π*) 모니터.
(4) B4 강baseline(NPE/flow, 결정론 iterative via master_train.py).
환경 conda activate vision_task, seed≥5 mean±std, figure PDF/metric JSON. 결정론 baseline은 개념적
collapse(평균=무정보)를 보이되 G0 non-attracting 병리는 전시 금지. 먼저 계획을 제시하고 시작하세요.
```

### ▶ Track T (companion 이론논문, JMLR) 프롬프트
```
prob_models(Paper 2) 프로젝트의 "companion 이론논문(JMLR) 트랙"을 진행합니다. 먼저
prob_models/paper/TRACKS.md의 §0과 §3(Track T), THEORY_PAPER_PLAN.md, theory_notes.tex를 읽으세요.

중요: root CLAUDE.md의 "prob_models out of scope"는 결정론 논문용이며 이 세션엔 적용 안 됨.
theory_notes.tex는 3차 정밀 리뷰를 통과한 정본이니 같은 엄밀도를 유지하세요(LaTeX 툴체인 없음 →
정적 검사). 일반성 유지(OGTT 특화 최소화).

목표(THEORY_PAPER_PLAN.md §9): (T2-1) Remark:nodouble에서 deferred된 ε_inc≤C(ε_H+ε_P)의 상수 C를
참 조건부 정칙성 하에서 실제 증명해 theory_notes.tex에 정리로 추가; (T2-2) approximate-MCMC 문헌
(Mitrophanov 2005, Rudolf–Schweizer 2018, Johndrow–Mattingly 등) 포지셔닝 절 + bib; (T2-3) 수치예시
E-a/b/c 설계; (T2-4) JMLR 논문 구조로 조립 + 표기 통일. 먼저 T2-1 증명 골격을 제시하고 시작하세요.
```

### ▶ Track W (본 논문 writing) 프롬프트
```
prob_models(Paper 2, A-DCVAE) 본 논문의 "writing/서사 트랙"을 진행합니다. 먼저
prob_models/paper/TRACKS.md의 §0과 §4(Track W), PAPER_BACKBONE.md, DISCUSSION.md(B1/B8/A''-scope)를
읽으세요.

중요: root CLAUDE.md의 "prob_models out of scope"는 결정론 논문용이며 이 세션엔 적용 안 됨. 이론은
축약본만 싣고 companion(\citep{adcvae-theory})을 인용하는 전략입니다(04_method §3.4·appendix 이미 반영).
LaTeX 툴체인 없음 → 정적 검사.

목표: (B1) intro 서사 — "공유 연산자 T, 무엇에 수렴하는지 고친다"; 존재(Thm A)·정확성(Thm B)를
전작(spectral norm+teacher forcing)과 평행하게; regression-to-mean을 '시작'으로. 결정론 방법은
baseline+브릿지(load-bearing 아님). (B8) related work — SBI/NPE, split-Gibbs/score-MCMC, dependency
networks. (W4) Table1 PICP↔MPIW 트레이드오프 서사·한계 정직성. Fig1 teaser 설계.
스코프: 일반 방법·OGTT는 한 예시, mDI는 부수적. 과대주장 금지. 먼저 intro 개요를 제시하고 시작하세요.
주의: experiment 서술·Table·Fig 최종본은 Track E 결과에 의존하므로, 결과 미확정 부분은 placeholder로.
```

### ▶ Track W 3차 (draft 마무리) 프롬프트  — 2026-07-19 이후
```
prob_models(Paper 2, A-DCVAE) writing 트랙의 마무리 단계입니다. 먼저 TRACKS.md §0·§4(Track W)와
로그(2026-07-19 항목), experiments/RESULTS.md를 읽으세요. 1·2차에서 B1/B8/W4/Fig1 서사 + Track E
확정결과(B3/B4/B7) 통합 + 이름 통일(A-DCVAE) + figure 9종 완비가 끝났습니다. 남은 일:

1. **Overleaf 첫 컴파일 (최우선, 구조 이슈 있음)**: main.tex가 경로를 두 기준으로 섞음 —
   `.sty`/`math_commands.tex`는 iclr2026/ 기준, `sections/`·`figures/`는 상위 paper/ 기준(심링크 X).
   그대로 올리면 Overleaf가 못 찾음. 해법: main.tex + iclr2026/*.sty + math_commands.tex를 sections/·
   figures/와 같은 레벨(paper/ 루트)로 재배치하거나, \input/\graphicspath 경로를 한 기준으로 통일.
   컴파일 후 실제 오류(정적검사로 못 잡는 것) 수정.
2. **bib 12개 서지 검증**: papamakarios2016fast·greenberg2019automatic·lueckmann2021benchmarking·
   papamakarios2021normalizing·ho2020denoising·song2021score·chung2023diffusion·kadkhodaie2021stochastic·
   vono2019split·coeurdoux2024plug·arnold2001conditionally·lipman2023flow 의 저자/venue/권/페이지 대조.
3. **통합 draft 전체 정독(coherence)**: Experiments가 커짐(non_identifiability→reference→baselines→
   ablation→predictive/noise). 흐름·전환·중복·섹션 순서 점검. §5 Analysis(Fig5)가 ablation 뒤에 오는
   배치가 자연스러운지 검토. abstract/intro가 주장 상향으로 밀도 높아짐 → 최종 tightening.
4. (선택) figure3_A(1D marginal) 본문 추가 여부, companion 이론논문(Track T) 진척과의 정합.

주의: 수치는 RESULTS.md가 정본(sliced-W2 0.170 vs 0.533 등). 과대주장 금지, 10k 스케일 역전은 유지.
figure 재생성이 필요하면 experiments/regen_figure3.py·fig1_teaser.py·phase4_figures.py 참고(메인
트리 CWD·GPU 필요). 먼저 Overleaf 구조 재배치안을 제시하고 합의 후 진행하세요.
```

### 로그
- 2026-07-13 — 최초 작성. 3트랙(E/T/W) 분리, todo·주의·kickoff 프롬프트 정리.
- 2026-07-14 — **Track W 1차 완료(B1/B8/W4/Fig1 서술).** intro를 "공유 연산자 T, 수렴 대상을
  고친다" 서사로 재작성(regression-to-mean=발단, decoupling=mixture identity로 유도, 결정론=baseline
  +극한, 존재/정확성=전작 수축+정렬의 확률론 평행). related work 신규(SBI·score-SDE·split-Gibbs/DPS·
  compatible conditionals). W4 정직성(PICP↔MPIW 방향성·Limitations/Sim2Real 완화). Fig1 캡션+설계스펙.
  bib 12개 추가(⚠️서지 검증). 미정의 `\xobs` 수정. 정적검사 통과(LaTeX 컴파일은 사용자 로컬).
  **미착수/의존**: experiment 수치·참조 posterior·ε_inc ablation·강baseline(B4)·Fig2/4/5는 Track E 대기.
- 2026-07-19 — **Track W 2차: Track E 확정결과 통합.** (0) main으로 sync(PR#2 merge, `54a3cb8`).
  (1) 캡션 3건 교정: Fig2(2패널→2×2 grid), Fig4(bimodal 과대주장→mDI fiber 6샘플·은닉 I 발산),
  Fig5(spiking→완만 단조 0.33→0.64). \fbox placeholder 4개를 실제 PDF `\includegraphics`로 교체
  (teaser·Fig2·Fig4·Fig5). (2) B4/B3/B7을 experiment1.tex 신규 3소절로 통합: `sec:reference`(B3
  참조 posterior, 과대분산 정직 갭), `sec:baselines`(B4 v2 캐노니컬 표+그림, sliced-W2 0.170 vs
  0.533), `sec:ablation`(B7 3보장 격리). (3) **주장 강도 상향+스케일 caveat**(Q1): abstract·
  contributions에 "캐노니컬 스케일에서 sliced-W2 우위" 반영, 10k 역전 명시. "synthetic systems"
  미실행 claim 삭제(정직성). (4) **method 이름 통일→A-DCVAE**(Dual CVAE/IterCVAEs/Iterative CVAEs
  전부 교체; figure3.py 라벨도 Iter→A-DCVAE 패치). (5) Fig1 teaser
  실제 PDF 생성(`figure1_teaser.pdf`, `experiments/fig1_teaser.py`, 개념도). (6) **figure3_B/C 재생성
  완료**(`experiments/regen_figure3.py`, canonical 체크포인트 seed=42, GPU1/메인트리 실행) — 범례
  Iter→A-DCVAE 통일, tail idx 319(S_I=1.454,σ=0.124)로 원본과 동일 샘플 재현. 정적검사 통과(cite·
  ref/label·env·이름잔재 0). **미완**: bib 12개 서지검증(draft 완성 후), 로컬 LaTeX 컴파일(서버에
  TeX 미설치 → tectonic 권장).
- 2026-07-14 — [Track T] T2-1 완료(`theory_notes.tex`: Thm:incbound + Lem:assembly + Ass:truereg +
  Rem:incconst). JMLR 스켈레톤 `paper/jmlr/main.tex` 신설(§5 구조 9절, 정리 stub은 theory_notes로
  [PORT] 포인터). T2-2 착수: §7 Positioning 초안 작성(approx-MCMC vs 우리 대상·ε_inc 인증서 차별화,
  ε_inc 보장/불보장 표 Table:einc) + bib 11종(Mitrophanov05, Rudolf–Schweizer18, Johndrow15,
  Medina16, Negrea21, Heckerman00, Arnold99, Diaconis99, Hairer11, Papamakarios16, Cranmer20).
  정적 검사 통과(env·cite/bibitem·ref/label). 다음: 정리 본문 PORT(T2-4) + T2-3 수치예시.
- 2026-07-14(2) — [Track T] T2-4 대부분 + T2-2 완료. `jmlr/main.tex`에 정리 본문·증명 이식(Thm A/rate/
  B/Prop inc/**Thm incbound**+Lem assembly full proof; 표준 예비지식은 인용 축약), 표기 bold θ,u 통일,
  **노테이션 표 Table:notation**(ε 5종 ε/ε_H/ε_P/ε_inc/ε_dist 구분) 추가, §7 Positioning+bib 17종.
  natbib+\intr 추가. 정적 검사 통과. 남음: T2-3(§8 수치 E-a/b/c), intro/discussion 산문, T2-5 제목.
- 2026-07-14(3) — [Track T] T2-3 설계 완료(`T2-3_NUMERICS_DESIGN.md`: E-a 선형가우시안 compatible→
  rate ρ²/exactness, E-b 비호환 dial δ→Thm B·incbound bound closed-form, E-c ab=c ridge→non-collapse·
  ε_inc≈0; 전부 Bures/grid closed-form, Track E B7/B3 공유). `main.tex` §8 산문 반영 + intro·discussion
  전면 작성. 정적 검사 통과(bibitem 전부 인용됨). 남음: T2-3 구현(Track E 공유), T2-5 제목.
- 2026-07-14(4) — [Track T] 사용자 결정: **T2-3 구현은 Track E 완료 후**로 연기. 사용자가 ~2026-08
  지역 응용수학 학회 발표(제목/초록 초안 작성) 예정 — Thm A+rate로 scope. 초록 리뷰에서 **과대주장 1건
  교정**: "noise = necessary condition for Doeblin minorization" → sufficient(+by-construction).
  (Rem:noise: 상수 디코더 δ_c도 minorization 만족 → necessary 아님. noise는 *절대연속* minorization에만
  necessary.) "breaking ergodicity" → 점질량 붕괴/ TV-uniform ergodicity 실패로 정밀화 권고.
- 2026-07-19 — [Track T] **T2-3 구현 완료**(`jmlr/numerics/`). Track E가 main 병합됨을 확인, T2-3은
  Track E와 독립(합성 toy, 학습 불필요)이라 착수. E-a/b/c 전부 이론 검증(κ=ρ²=0.640, Thm B·incbound 전
  δ 성립, coverage 0.95, along-ridge 1.19 vs 0.000, ε_inc 0.008). 그림 3종 main.tex §8 embed(+graphicx).
  `.gitignore`에 jmlr/figures/*.pdf negation 추가. 정적 검사 통과. **주의: 이 브랜치는 b20c3aa 분기라
  main(8e57250, Track E/W 병합됨)보다 뒤처짐 → push/PR 전 rebase 필요**(TRACKS.md 양쪽 수정 충돌 가능).
- 2026-07-19(2) — [Track T] main으로 rebase 완료(TRACKS.md 로그 충돌 1건, 시간순 병합으로 해결) →
  push → PR #4 merge 완료(main `97dad95`). 이어서 **T2-4 마무리 + T2-5 동시 진행**: (1) §3에 N1–N4
  노이즈 4종 taxonomy 문단 신설(N1→rate, N2→ε_H/ε_P, N3/N4→spread·minorization·커플링 상쇄, Rem:noises
  연결). (2) §9 discussion에 Track E 실측 연결 문단(κ 0.406→0.709 N1-off, ε_inc≈0.029, autocorr 0.012;
  bib `adcvae-application` 신설로 응용논문 상호인용 완성). (3) T2-5: 제목 후보 6개 제시, **사용자가
  현재 초안 유지 결정**. (4) 부수 수정: `iclr2026_conference.bib`의 `adcvae-theory` 제목이 확정 제목과
  달랐던 것(옛 후보 문구) 동기화. 정적 검사 통과(bibitem 전부 인용됨, env·ref/label 정합).
