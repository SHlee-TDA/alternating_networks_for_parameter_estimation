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
- [~] **E2 (B4): 강baseline** — 10k-scale 완료 + **50k 캐노니컬 스케일 재현 진행중**(2026-07-14).
  self-contained conditional RealNVP NPE 단일망 + 결정론 MSE regressor(collapse: along-fiber std 0).
  **결정론 iterative는 iter_det 체크포인트 대신 깨끗이 수렴하는 MSE regressor 사용** — G0 병리 전시
  회피(§주의).
  - 10k-scale 결과: decoupling이 단일망 NPE를 명확히 이기지 못함(sliced-W2 무승부, coverage는 NPE 우위).
  - **50k 캐노니컬(v1) 결과: 순위가 뒤집힘**(A-DCVAE가 sliced-W2·mDI 방향 모두 우위) — 단
    ⚠️ **confound 발견**: NPE-flow/det-reg가 patience=40(하드코딩)으로 학습, A-DCVAE만 patience=200
    (config 기본값)을 받음. 2026-07-14 `b4_baselines.py`에 `--patience` 인자 추가해 세 방법 모두
    통일, **v2(patience=200 전체 통일) 캐노니컬 재실행 중**(백그라운드, PID 기록은 세션 로그 참조).
    **v2 완료 전까지 "decoupling이 이긴다/못이긴다" 어느 쪽도 논문에 확정 서술 금지.**
  - `figure_b4_baselines.pdf` (v1 반영, v2 완료 시 갱신 예정). **(정규화기 불일치 버그는 별도로 이미
    수정됨: 모든 방법을 동일 컨텍스트에서 학습).**
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
- [ ] **T2-1: `ε_inc ≤ C(ε_H+ε_P)` 상수 실제 증명** (현재 `Remark:nodouble`에서 deferred). 참
  조건부 정칙성 하에서 `C=C(κ,L_H,L_P)` 명시 → `theory_notes.tex`에 정리로 추가.
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

### 로그
- 2026-07-13 — 최초 작성. 3트랙(E/T/W) 분리, todo·주의·kickoff 프롬프트 정리.
- 2026-07-14 — **Track W 1차 완료(B1/B8/W4/Fig1 서술).** intro를 "공유 연산자 T, 수렴 대상을
  고친다" 서사로 재작성(regression-to-mean=발단, decoupling=mixture identity로 유도, 결정론=baseline
  +극한, 존재/정확성=전작 수축+정렬의 확률론 평행). related work 신규(SBI·score-SDE·split-Gibbs/DPS·
  compatible conditionals). W4 정직성(PICP↔MPIW 방향성·Limitations/Sim2Real 완화). Fig1 캡션+설계스펙.
  bib 12개 추가(⚠️서지 검증). 미정의 `\xobs` 수정. 정적검사 통과(LaTeX 컴파일은 사용자 로컬).
  **미착수/의존**: experiment 수치·참조 posterior·ε_inc ablation·강baseline(B4)·Fig2/4/5는 Track E 대기.
