# Companion Theory Paper — Plan (Target: JMLR)

> 이 문서는 `theory_notes.tex`의 이론을 **독립 논문**으로 발전시키기 위한 계획서다. 응용(OGTT/
> A-DCVAE) 논문과 분리하고, 상호 인용한다. 최초 작성 2026-07-13.
> 관련: [[PAPER_BACKBONE.md]], [[DISCUSSION.md]], `theory_notes.tex`.

---

## 0. 한 줄 논지
연속 상태공간에서 **독립적으로 학습된 신경망 조건부**로 구성한 amortized 교대(pseudo-Gibbs)
샘플러의 **수렴·rate·일관성(steering)**에 대한 최초의 엄밀한 이론. 핵심 신기여는 계산 가능한
**self-consistency 인증서 `ε_inc`**와, 존재(TV)/rate(W_2)/steering의 통합 처리.

## 1. Target venue
- **1순위: JMLR.** 근거: 방법론이 ML 태생, `ε_inc`가 실무 진단(진단 통계)으로 이어져 이론+
  재현코드+수치검증을 엮기 좋음. JMLR은 SOTA를 요구하지 않고 **엄밀성·명료성·완결성**을 보상.
- **백업: SIAM J. Mathematics of Data Science (SIMODS).** 정리 중심·수학적 엄밀성 톤.
- **지양**: 순수확률(Bernoulli/SPA — 확률론 자체 신규성 요구, 과함), SIAM JUQ(전작 SIAM 계열과
  중복 우려).

## 2. 정직한 신규성·포지셔닝 (리뷰어 방어의 핵심)
개별 도구는 표준(리뷰어가 반드시 지적): Doeblin minorization, iterated random functions의 W_2
수축(Diaconis–Freedman), 불변측도 perturbation(Mitrophanov). **따라서 신규성은 다음에 둔다:**
1. **대상의 신규성**: 학습된, 잠재적으로 **비호환(incompatible)** dual conditional을 연속공간에서
   엄밀히 다룸. Heckerman(2000)은 **이산 + 점근**만; 우리는 **연속 + 정량**.
2. **`ε_inc` (계산 가능 self-consistency 인증서)**: 참을 모르고도 측정 가능한, 근사오차의
   "관측 가능한 그림자". (진짜 새 물건.) `ε_inc=W_2(Π→,π*)`로 단일 run에서 추정.
3. **통합 프레임 + 노이즈 taxonomy**: 존재는 노이즈(N4)+compact로 가정 없이(Thm A), rate는
   조건-Lipschitz(N1)로(W_2 수축), steering은 perturbation(Thm B). "수축↔붕괴 역설"을 4-노이즈
   분리로 해소.

## 3. 반드시 걸쳐야 할 문헌 (이론판 "현대 이웃" — 안 걸치면 치명적)
- **Approximate / perturbed MCMC** (Thm B가 이 한복판): Mitrophanov (2005, perturbation bounds),
  Rudolf–Schweizer (2018, W-perturbation), Johndrow–Mattingly (approx. MCMC), Medina-Aguayo et al.
  (noisy/Monte-Carlo-within-Metropolis), Negrea–Rosenthal. → 우리 차별점: 학습된 dual conditional
  + `ε_inc` + 노이즈 taxonomy.
- **Iterated random functions / Wasserstein ergodicity**: Diaconis–Freedman (1999), Hairer–Mattingly,
  Ollivier (Ricci curvature). → sweep 수축의 계보.
- **Dependency networks / compatible conditional specification**: Heckerman et al. (2000),
  Arnold–Castillo–Sarabia. → `ε_inc`·compatibility의 계보.
- **Amortized / simulation-based inference**: Papamakarios–Murray, Greenberg, Cranmer et al.
- **Denoising = score**: Vincent (2011), Song et al. — (배경; `ε_H,ε_P`가 왜 작아지는가의 해석.)

## 4. Standalone bar를 넘기 위해 닫아야 할 3가지 (현 노트의 공백)
1. **`ε_inc ≤ C(ε_H+ε_P)` 상수 실제 증명** (현재 `Remark:nodouble`에서 deferred). 참 조건부의
   정칙성 가정 하에서 `C=C(κ,L_H,L_P)`를 명시. — **이론 정리 1개 추가.**
2. **정량적 rate + 수치 예시** — κ<1이 실제로 성립하고 bound가 의미 있음을 보이는 최소 실험.
3. **approximate-MCMC 포지셔닝 절** 신설(§3의 문헌 대비).

## 5. 제안 논문 구조 (JMLR)
1. **Introduction** — amortized pseudo-Gibbs 문제; 왜 연속 확장이 비자명(Heckerman 이산 vs 연속);
   3 기여(존재/rate/steering + `ε_inc`).
2. **Preliminaries** — 측도·Markov kernel·TV·W_2·Doeblin·Ionescu–Tulcea (노트 §1–3 축약).
3. **The amortized pseudo-Gibbs kernel** — 학습된 dual conditional로 만든 커널의 형식화(노트 §4).
4. **Existence & uniform ergodicity** (Theorem A) — 노트 §5–8.
5. **Mixing rate: Wasserstein contraction** — 노트 §9–10.
6. **Steering & consistency** (Theorem B + `ε_inc` + **새 상수 정리**) — 노트 §11–13 + §4-공백1.
7. **Positioning vs approximate MCMC** — §3 문헌 대비.
8. **Numerical illustration** — §6 아래.
9. **Discussion / limitations** — (A-contraction) 가정의 실질성, σ-floor, 확장.

## 6. 수치 예시 계획 (theory 검증용, 응용과 분리)
"이론이 실제로 작동함"을 보이는 최소·통제된 실험:
- **(E-a) Linear-Gaussian compatible**: 조건부가 선형가우시안이라 참 posterior·κ·ε 모두 closed-form.
  → Corollary B1(정확성) + rate `κ^n` 검증.
- **(E-b) Controlled incompatibility**: 한 조건부에 의도적 섭동을 넣어 `ε_inc>0`을 dial. → Theorem B
  상한이 실제 `W_2(ν*,p_true)`를 bound하는지, `ε_inc`가 `ε_app`을 어떻게 추적하는지 검증.
- **(E-c) Non-identifiable toy**: 곱-축퇴(예: `a·b=c`) 저차원 ODE/사상. → 붕괴 없는 support 복원 +
  `ε_inc` 모니터. (OGTT의 `S_I·σ` 구조의 순수-수학 축소판, 응용논문과 중복 회피.)

## 7. 응용 논문(A-DCVAE)과의 분업
| | Companion 이론 논문 (JMLR) | 응용 논문 (ICLR/AISTATS/TMLR) |
|---|---|---|
| 대상 | 일반 amortized pseudo-Gibbs | A-DCVAE + OGTT 비식별 |
| 정리 | 전체 진술 + 증명 | **축약 진술 + 직관 + companion 인용** |
| 실험 | 통제된 이론검증(E-a/b/c) | OGTT posterior·baseline·ablation |
| 상호인용 | 응용을 motivating example로 | 이론을 [Anon, companion]으로 |
- **중복/self-plagiarism 회피**: 이론 논문은 OGTT-특화 서술 최소화(일반 프레임), 응용 논문은 증명
  중복 없이 인용.

## 8. 리스크 & 리뷰어 반론 대비
- "기법이 표준" → §2 신규성(대상·`ε_inc`·통합)으로 방어 + §3 포지셔닝.
- "(A-contraction) 가정이 강함/검증불가" → 정직히 가정으로 두고, `ε_inc`·autocorrelation로 **경험적
  검증**(E-b) + condition-noise가 촉진함을 Bishop로 연결. κ를 자유 파라미터로.
- "`ε_inc`가 유용한가" → 진단 통계로서 계산 가능·단일 run 추정(`=W_2(Π→,π*)`)임을 E-b로 실증.
- "σ-floor로 정확 복원 불가" → non-identifiable ridge 폭 대비 작음(E-c), 그리고 애초에 point가 아닌
  분포 복원이 목표.

## 9. 작업 목록 (제안 순서)
- [ ] T2-1. `ε_inc` 상수 정리 증명(§4-공백1) → `theory_notes.tex`에 추가.
- [ ] T2-2. approximate-MCMC 포지셔닝 절 초안(§7) + bib 확충.
- [ ] T2-3. 수치 예시 E-a/b/c 설계·구현·plot.
- [ ] T2-4. `theory_notes.tex` → JMLR 논문 초안 조립(§5 구조), 표기 통일(plain/bold) 포함.
- [ ] T2-5. 제목 확정(후보: *"Convergence and Consistency of Amortized Pseudo-Gibbs Samplers with
      Learned Conditionals"*, *"Ergodicity and Steering of Alternating Denoising Samplers"*).

### 로그
- 2026-07-13 — 최초 작성. JMLR 1순위 확정. standalone 3-공백(ε_inc 상수·approx-MCMC 포지셔닝·
  수치검증), 구조·분업·리스크 정리.
