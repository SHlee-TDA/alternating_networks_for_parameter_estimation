import os
import subprocess
import time

DATASETS = ["sir", "lotka_volterra", "ogtt_simul"]
# True면 Single Baseline, False면 Iterative (config.py 기준)
DETERMINISTIC_MODELS = {"single_det": "True", "iter_det": "False"} 
PROBABILISTIC_MODELS = {"single_cvae": "True", "iter_cvae": "False"}

def run_experiment(script_name, dataset, is_baseline, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    
    # Argparse를 통해 Config 값을 덮어쓰는 명령어
    cmd = [
        "python", script_name,
        "--system", dataset,
        "--run_baseline", is_baseline,
        "--results_dir", save_dir
    ]
    
    log_file = os.path.join(save_dir, "training.log")
    with open(log_file, "w") as f:
        print(f"▶ 실행 중: {' '.join(cmd)}")
        start_time = time.time()
        process = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
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
            save_dir = os.path.join("checkpoints", "deterministic", dataset, model_name)
            # 루트 디렉토리의 main.py 실행
            run_experiment("main.py", dataset, is_baseline, save_dir)
            
        # 2. Probabilistic Models (Paper 2)
        for model_name, is_baseline in PROBABILISTIC_MODELS.items():
            save_dir = os.path.join("checkpoints", "probabilistic", dataset, model_name)
            # 범용화된 확률론적 스크립트 실행
            run_experiment("run_cvae_master.py", dataset, is_baseline, save_dir)

if __name__ == "__main__":
    main()