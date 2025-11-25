# TODO: Refactoring & Optimization Strategy

- [ ] **Dependency Injection (Priority: Medium)**
    - `systems/ogtt_simul.py`: 전역 변수(`SIGMA_T_POINTS`, `SIGMA_G_T` 등)를 제거하고, `OgttSimul` 클래스 초기화 시 파라미터로 주입받도록 수정. (실험 재현성 확보)
- [ ] **Numerical Precision (Priority: Low)**
    - `analysis/calibrate_sde_params.py`: 30분 간격의 선형 보간(`np.interp`) 대신 `Cubic Spline` 도입 고려. (SDE Solver의 미분 가능성 확보)
- [ ] **Sampling Efficiency (Priority: Low)**
    - `data_loader.py`: `sample_from_lognorm` 내의 Rejection Sampling 로직을 최적화하거나 분포 파라미터를 재검토.