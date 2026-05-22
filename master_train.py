import os
import subprocess
import time

DATASETS = ["ogtt_simul"]
# True면 Single Baseline, False면 Iterative (config.py 기준)
DETERMINISTIC_MODELS = {"single_det": "True", "iter_det": "False"} 
PROBABILISTIC_MODELS = {"single_cvae": "True", "iter_cvae": "False"}

def run_experiment(script_or_module, dataset, is_baseline, save_dir, is_module=False):
    os.makedirs(save_dir, exist_ok=True)
    
    # -m 플래그를 사용하여 모듈로 실행할지 여부 결정 (Import 경로 꼬임 방지)
    if is_module:
        cmd = ["python", "-m", script_or_module]
    else:
        cmd = ["python", script_or_module]
        
    cmd.extend([
        "--system", dataset,
        "--run_baseline", is_baseline,
        "--results_dir", save_dir
        # 명시적으로 --epochs를 주지 않으므로, 각 Config에 설정된 기본값(예: 1000, 3000)으로 풀-트레이닝 됩니다.
    ])
    
    env = os.environ.copy()
    env["TQDM_DISABLE"] = "1"      # tqdm 프로그레스 바 완전 비활성화 (로그 파일 테러 방지)
    env["PYTHONUNBUFFERED"] = "1"  # 버퍼링 비활성화 (tail -f 로 실시간 모니터링 가능하게 함)
    
    log_file = os.path.join(save_dir, "training.log")
    with open(log_file, "w") as f:
        print(f"▶ 실행 중: {' '.join(cmd)}")
        print(f"  (💡 모니터링 명령어: tail -f {log_file})")
        start_time = time.time()
        
        # subprocess.run은 각 학습이 끝날 때까지 대기하며, 종료 시 GPU VRAM을 완전히 반환합니다 (OOM 방지)
        process = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
        elapsed = time.time() - start_time
        
        if process.returncode == 0:
            print(f"✅ 완료! (소요시간: {elapsed:.2f}초) -> {save_dir}\n")
        else:
            print(f"❌ 실패! 로그 확인: {log_file}\n")

def main():
    print("🚀 Master Training Pipeline 시작 (총 12세션)\n")
    
    for dataset in DATASETS:
        # 1. Deterministic Models (Paper 1)
        for model_name, is_baseline in DETERMINISTIC_MODELS.items():
            save_dir = os.path.join("results", "master_train" , "deterministic", dataset, model_name)
            # 루트 디렉토리의 main.py는 일반 스크립트로 실행
            run_experiment("main.py", dataset, is_baseline, save_dir, is_module=False)
            
        # 2. Probabilistic Models (Paper 2)
        for model_name, is_baseline in PROBABILISTIC_MODELS.items():
            save_dir = os.path.join("results", "master_train" , "probabilistic", dataset, model_name)
            # prob_models/main.py 는 경로 에러 방지를 위해 반드시 모듈(-m)로 실행!
            run_experiment("prob_models.main", dataset, is_baseline, save_dir, is_module=True)

if __name__ == "__main__":
    main()