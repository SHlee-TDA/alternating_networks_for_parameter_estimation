import os
import random
import numpy as np
import torch
import importlib
import copy # config 객체를 복사하기 위해 추가

from config import Config
from data_loader import DataGenerator, create_dataloaders # check_lagrangian_applied는 삭제 가능
from models import HiddenVarPredictor, ParameterEstimator
from trainer import Trainer
from analyzer import Analyzer
from utils import Normalizer

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_system_class(system_name):
    """
        문자열 이름으로부터 시스템 클래스를 동적으로 임포트합니다.
    """
    module_name = f"systems.{system_name}"
    class_name = "".join(word.capitalize() for word in system_name.split('_'))
    module = importlib.import_module(module_name)
    return getattr(module, class_name)

# [수정] run_experiment의 시그니처가 변경되었습니다.
#         이제 'base_config' (데이터 생성 시 사용된 config)를 받습니다.
def run_experiment(exp_config, system, data_tuple, normalizer, base_config):
    """단일 실험을 실행하는 함수"""
    
    # [수정] config 객체를 새로 만드는 대신,
    #         데이터 생성에 사용된 base_config를 깊은 복사(deepcopy)하여 시작합니다.
    #         (참고: config 클래스가 복잡하지 않다면 얕은 복사(copy.copy)도 가능)
    config = copy.deepcopy(base_config) 
    
    # 실험별 설정으로 덮어쓰기
    config.EXPERIMENT_NAME = exp_config['name']
    config.USE_SPECTRAL_NORM = exp_config['use_spectral_norm']
    config.USE_CONSISTENCY_LOSS = exp_config['use_consistency_loss']
    
    # [중요] config.USE_LAGRANGIAN은 덮어쓰지 *않습니다*.
    #         base_config에 이미 올바른 값(data_tuple과 일치하는)이 들어있습니다.
    #         config.py의 EXPERIMENTS 딕셔너리에 있는 'use_lagrangian' 키는
    #         데이터를 선택(캐싱)하는 용도로만 사용됩니다.
    
    print(f"\n===== Running Experiment: {config.EXPERIMENT_NAME} on {system.name.upper()} =====")
    print(f"Configuration Settings Check: ")
    print(f"  USE_SPECTRAL_NORM: {config.USE_SPECTRAL_NORM}")
    print(f"  USE_CONSISTENCY_LOSS: {config.USE_CONSISTENCY_LOSS}")
    print(f"  USE_LAGRANGIAN: {config.USE_LAGRANGIAN} (From Data Source)") # <-- 이제 정확함
    
    
    # 이제 data_tuple과 config.USE_LAGRANGIAN 설정이 100% 일치합니다.
    train_loader, val_loader, test_loader, p_initial_guess = create_dataloaders(data_tuple, config)
    
    try:
        x_sample, y_sample, p_sample = next(iter(train_loader))
    except StopIteration:
        raise ValueError("DataLoader is empty. Cannot determine model dimensions.")

    FLAT_X_DIM = x_sample.shape[1]
    FLAT_Y_DIM = y_sample.shape[1]
    NUM_PARAMS = p_sample.shape[1]

    print(f"Model dimensions determined from data:")
    print(f"  flat_x_dim: {FLAT_X_DIM}")
    print(f"  flat_y_dim: {FLAT_Y_DIM}")
    print(f"  num_params: {NUM_PARAMS}")
        
    
    f_theta = HiddenVarPredictor(
        flat_x_dim=FLAT_X_DIM,
        flat_y_dim=FLAT_Y_DIM,
        num_params=NUM_PARAMS,
        model_config=config.MODEL_CONFIG['f_theta'],
        use_spectral_norm=config.USE_SPECTRAL_NORM,
        initialization_config=config.MODEL_CONFIG.get('initialization')
    ).to(config.DEVICE)
    
    g_phi = ParameterEstimator(
        flat_x_dim=FLAT_X_DIM,
        flat_y_dim=FLAT_Y_DIM,
        num_params=NUM_PARAMS,
        model_config=config.MODEL_CONFIG['g_phi'],
        use_spectral_norm=config.USE_SPECTRAL_NORM,
        initialization_config=config.MODEL_CONFIG.get('initialization')
    ).to(config.DEVICE)
    
    trainer = Trainer(f_theta, g_phi, 
                      train_loader, val_loader,
                      config, normalizer)
    
    # 학습
    f_theta, g_phi, history = trainer.train()
    
    # 분석
    analyzer = Analyzer(f_theta, g_phi, test_loader, config, system, p_initial_guess, normalizer, history)
    analyzer.plot_loss_curves()
    analyzer.analyze_spectral_norms()
    p_true, p_pred = analyzer.evaluate_predictions()
    analyzer.plot_scatter(p_true, p_pred)
    analyzer.plot_spectral_norms_by_layer()
    analyzer.plot_phase_portraits()

def main():
    """메인 실행 함수"""
    # config.py에서 정의된 기본 설정을 로드합니다.
    global_config = Config()
    set_seed(global_config.SEED)
    
    # 1. 시스템 로드
    SystemClass = get_system_class(global_config.SYSTEM_NAME)
    system = SystemClass()
    normalizer = Normalizer(system, global_config.DEVICE)

    # 2. [수정] 데이터 캐시 준비
    data_cache = {} # 키: (bool), 값: (data_tuple, config_used)

    # 2a. [수정] EXPERIMENTS 리스트를 먼저 순회하며 필요한 데이터 종류를 파악
    required_lagrangian_settings = set()
    for exp_config in global_config.EXPERIMENTS:
        use_lagrangian = exp_config.get('use_lagrangian', False)
        required_lagrangian_settings.add(use_lagrangian)

    print(f"Required data versions: {required_lagrangian_settings}")

    # 2b. [수정] 필요한 데이터 버전별로 데이터 생성 및 캐싱
    for use_lagrangian in required_lagrangian_settings:
        data_key = use_lagrangian # 키는 그냥 boolean 값 (True/False)
        
        print(f"\n--- Generating data for USE_LAGRANGIAN = {use_lagrangian} ---")
        
        # 이 데이터 생성을 위한 전용 config 생성
        # global_config를 깊은 복사하여 사용
        data_gen_config = copy.deepcopy(global_config) 
        data_gen_config.USE_LAGRANGIAN = use_lagrangian
        # config.py 템플릿의 'USE_LAGRANGIAN'은 이제 무시됩니다.
        # 여기서 설정한 값이 DataGenerator로 전달됩니다.

        data_gen = DataGenerator(system, data_gen_config)
        data_tuple = data_gen.generate_data()
        
        # 데이터와, 이 데이터를 생성할 때 쓴 config를 함께 캐싱
        data_cache[data_key] = (data_tuple, data_gen_config)
        print(f"--- Data generation for {use_lagrangian} complete. ---")


    # 3. 정의된 모든 실험 실행
    print(f"\n--- Starting {len(global_config.EXPERIMENTS)} experiments ---")
    for exp_config in global_config.EXPERIMENTS:
        
        # 이 실험에 필요한 데이터 키를 찾음
        data_key = exp_config.get('use_lagrangian', False)
        
        # 캐시에서 올바른 데이터와 해당 데이터의 config를 로드
        data_tuple, base_config = data_cache[data_key]
        
        # 실험 함수에 (실험설정, 시스템, 데이터, 생성용config) 전달
        run_experiment(exp_config, system, data_tuple, normalizer, base_config)

if __name__ == '__main__':
    main()