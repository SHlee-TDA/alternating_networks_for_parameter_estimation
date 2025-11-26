import subprocess
import sys
import time
import os
import torch
import copy
import importlib

from config import Config
from data_loader import DataGenerator
from utils import Normalizer

def get_system_class(system_name):
    """시스템 클래스 동적 로딩 (main.py와 동일 로직)"""
    module_name = f"systems.{system_name}"
    class_name = "".join(word.capitalize() for word in system_name.split('_'))
    try:
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    except Exception as e:
        print(f"[Error] Failed to load system '{system_name}': {e}")
        sys.exit(1)

def prepare_datasets():
    """
    [Phase 5 핵심] 모든 실험에 필요한 데이터셋을 미리 생성/확인합니다.
    이 과정이 완료되면, 개별 실험(main.py)들은 시뮬레이션 없이 즉시 데이터를 로드합니다.
    """
    print(f"\n{'='*60}")
    print(f"📦 [Step 1] Preparing Datasets (Pre-caching)")
    print(f"{'='*60}")

    # 1. System 초기화
    SystemClass = get_system_class(Config.SYSTEM_NAME)
    system = SystemClass()
    
    # SDE Scaling 적용 (데이터 생성 시 중요)
    if hasattr(Config, 'SDE_SCALE_FACTORS'):
        if hasattr(system, 'bias_scale'):
            system.bias_scale = Config.SDE_SCALE_FACTORS.get('bias_scale', 1.0)
        if hasattr(system, 'diffusion_scale'):
            system.diffusion_scale = Config.SDE_SCALE_FACTORS.get('diffusion_scale', 1.0)

    # 2. 필요한 데이터 버전 파악
    # (SDE 여부, Lagrangian 여부)의 유니크한 조합을 찾음
    required_versions = set()
    for exp in Config.EXPERIMENTS:
        use_sde = exp['use_sde']
        use_lag = exp.get('use_lagrangian', Config.USE_LAGRANGIAN)
        required_versions.add((use_sde, use_lag))
    
    print(f"Found {len(required_versions)} unique dataset configurations: {required_versions}")

    # 3. 데이터 생성 (또는 확인)
    for (use_sde, use_lag) in required_versions:
        mode_str = "SDE" if use_sde else "ODE"
        print(f"\n>> Checking Data: Mode={mode_str}, Lagrangian={use_lag}")
        
        # 임시 Config 생성
        gen_config = copy.deepcopy(Config)
        gen_config.USE_SDE = use_sde
        gen_config.USE_LAGRANGIAN = use_lag
        
        # DataGenerator 인스턴스화
        data_gen = DataGenerator(system, gen_config)
        
        # 파일 존재 여부는 generate_data() 내부에서 체크함
        # 없으면 생성하고 저장, 있으면 로드 (시뮬레이션 Skip)
        _ = data_gen.generate_data()
        
    print(f"\n✅ All datasets are ready on disk. Proceeding to experiments.\n")


def run_experiments():
    """
    [Phase 5 핵심] 준비된 데이터를 사용하여 실험을 순차 수행합니다.
    """
    experiments = Config.EXPERIMENTS
    total_exps = len(experiments)
    
    # 실행 환경 확인
    device_status = "CUDA (GPU)" if torch.cuda.is_available() else "CPU"
    print(f"{'='*60}")
    print(f"🚀 [Step 2] Starting Phase 5 Execution")
    print(f"📍 Total Experiments: {total_exps}")
    print(f"📍 Environment: {device_status}")
    print(f"{'='*60}\n")
    
    fail_count = 0
    
    for i, exp_config in enumerate(experiments):
        exp_name = exp_config['name']
        print(f"\n>>> [{i+1}/{total_exps}] Running Experiment: {exp_name}")
        
        # 중복 실행 방지 (Result 폴더 확인)
        # ExperimentLogger가 사용하는 경로: results/{SYSTEM_NAME}/{EXP_NAME}
        result_dir = os.path.join(Config.RESULTS_DIR, Config.SYSTEM_NAME, exp_name)
        # 완료된 실험인지 확인 (final_metrics.json 존재 여부)
        if os.path.exists(os.path.join(result_dir, 'final_metrics.json')):
            print(f"  ✅ Skipping: Results already exist in {result_dir}")
            continue
            
        # main.py 실행 커맨드 구성
        cmd = [
            sys.executable, "main.py",
            "--name", exp_name,
            "--use_sde", str(exp_config['use_sde']),
            "--scenario", str(exp_config['scenario']),
            "--real_ratio", str(exp_config.get('real_ratio', 0.0)),
            "--val_source", str(exp_config.get('val_source', 'sim')),
            "--use_spectral_norm", str(exp_config.get('use_spectral_norm', False)),
            "--use_consistency_loss", str(exp_config.get('use_consistency_loss', False))
        ]
        
        # Subprocess 실행
        start_time = time.time()
        try:
            # check=True: 에러 발생 시 CalledProcessError 발생
            subprocess.run(cmd, check=True)
            
            elapsed = time.time() - start_time
            print(f"  ✨ Finished in {elapsed/60:.1f} minutes.")
            fail_count = 0 # 성공 시 카운트 초기화
            
        except subprocess.CalledProcessError as e:
            print(f"  ❌ [ERROR] Experiment {exp_name} failed (Code: {e.returncode})")
            fail_count += 1
            
            # 3번 연속 실패 시 전체 중단 (안전장치)
            if fail_count >= 3:
                print("\n🚨 [CRITICAL] Too many consecutive failures. Aborting.")
                sys.exit(1)
        except KeyboardInterrupt:
            print("\n🛑 Interrupted by user.")
            sys.exit(0)
            
        time.sleep(5) # GPU 메모리 반환 대기

    print(f"\n{'='*60}")
    print(f"🎉 Phase 5 Complete!")
    print(f"{'='*60}")

if __name__ == "__main__":
    # 1. 데이터셋 선행 준비 (캐싱 보장)
    prepare_datasets()
    
    # 2. 실험 순차 실행
    run_experiments()