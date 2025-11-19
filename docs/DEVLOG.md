# 개발 로그 (Development Log)

## 2025-11-19: NIH 데이터 로더 구현 및 수치해석 이슈 해결 (Final)

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