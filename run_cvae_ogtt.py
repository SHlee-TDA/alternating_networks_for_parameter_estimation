import os
import torch
import numpy as np
import json
import argparse
from torch.utils.data import TensorDataset, DataLoader, random_split, Subset

# 1. 시스템 및 유틸리티 (기존 자원 100% 재사용)
from systems.ogtt_simul import OgttSimul
from data_loader import DataGenerator, RealOGTTDataLoader
from utils import Normalizer
from tools.exp_tools import set_seed
from tools.interactive_file_selector import interactive_file_selector

# 2. 확률론적 패러다임 전용 모듈
from prob_models.config import ProbConfig
from prob_models.models import HiddenStateCVAE, ParameterCVAE
from prob_models.trainer import ProbabilisticTrainer
from prob_models.analysis import run_full_analysis

def main():
    parser = argparse.ArgumentParser(description="Run Probabilistic CVAE Pipeline")
    parser.add_argument('--load', action='store_true', help="Skip training and load pre-trained weights")
    args = parser.parse_args()
    
    print("\n\033[1;35m========================================================\033[0m")
    print("\033[1;35m  Probabilistic Pipeline: Alternating CVAEs for OGTT  \033[0m")
    print("\033[1;35m========================================================\033[0m")
    
    config = ProbConfig()
    set_seed(config.SEED)
    system = OgttSimul()
    
    # =================================================================
    # Phase 1: Data Generation & Loading
    # =================================================================
    print(f"\n[Phase 1] Preparing Simulation Data...")
    data_gen = DataGenerator(system, config)
    obs_sim, hid_sim, params_sim, _ = data_gen.generate_data()
    
    # Normalizer 세팅 (기존 코드와 완벽 호환)
    scale_obs = np.percentile(np.abs(obs_sim), 99.9)
    scale_hid = np.percentile(np.abs(hid_sim), 99.9)
    p_min, p_max = np.min(params_sim, axis=0), np.max(params_sim, axis=0)
    
    normalizer = Normalizer(
        system, config.DEVICE, 
        state_scales=[scale_obs * 1.2, scale_hid * 1.2], 
        param_bounds=(p_min / 1.2, p_max * 1.2), 
        use_log_params=config.USE_LOG_PARAMS
    )
    
    # 텐서 변환 및 정규화
    X_sim = normalizer.normalize_inputs(torch.FloatTensor(obs_sim).view(len(obs_sim), -1).to(config.DEVICE), 'observed')
    Y_sim = normalizer.normalize_inputs(torch.FloatTensor(hid_sim).view(len(hid_sim), -1).to(config.DEVICE), 'hidden')
    P_sim = normalizer.normalize_params(torch.FloatTensor(params_sim).to(config.DEVICE))
    
    dataset_sim = TensorDataset(X_sim, Y_sim, P_sim)
    
    # Train / Val / Test 분할
    test_len = int(len(dataset_sim) * config.TEST_SPLIT)
    val_len = test_len
    train_len = len(dataset_sim) - val_len - test_len
    
    train_set, val_set, test_set = random_split(
        dataset_sim, [train_len, val_len, test_len],
        generator=torch.Generator().manual_seed(config.SEED)
    )
    
    train_loader = DataLoader(train_set, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=config.BATCH_SIZE, shuffle=False)
    
    # =================================================================
    # Phase 2: Model Architecture Setup
    # =================================================================
    flat_x_dim = X_sim.shape[1]
    flat_y_dim = Y_sim.shape[1]
    flat_theta_dim = P_sim.shape[1]
    
    print(f"\n[Phase 2] Initializing Probabilistic CVAE Networks...")
    print(f" - Input X dim: {flat_x_dim} | Y dim: {flat_y_dim} | Theta dim: {flat_theta_dim}")
    
    hidden_cvae = HiddenStateCVAE(
        x_dim=flat_x_dim, theta_dim=flat_theta_dim, y_dim=flat_y_dim, 
        latent_dim=config.LATENT_DIM_HIDDEN, hidden_dims=config.HIDDEN_DIMS
    ).to(config.DEVICE)
    
    param_cvae = ParameterCVAE(
        x_dim=flat_x_dim, y_dim=flat_y_dim, theta_dim=flat_theta_dim, 
        latent_dim=config.LATENT_DIM_PARAM, hidden_dims=config.HIDDEN_DIMS
    ).to(config.DEVICE)
    
    # =================================================================
    # Phase 3: Independent Training
    # =================================================================
    save_dir = os.path.join(config.RESULTS_DIR, config.EXPERIMENT_NAME)
    hnet_path = os.path.join(save_dir, 'hidden_cvae.pth')
    pnet_path = os.path.join(save_dir, 'param_cvae.pth')

    if args.load and os.path.exists(hnet_path) and os.path.exists(pnet_path):
        print(f"\n\033[1;32m[Phase 3] --load flag detected. Skipping training...\033[0m")
        print(f"Loading weights from:\n - {hnet_path}\n - {pnet_path}")
        
        hidden_cvae.load_state_dict(torch.load(hnet_path, map_location=config.DEVICE))
        param_cvae.load_state_dict(torch.load(pnet_path, map_location=config.DEVICE))
        hidden_cvae.eval()
        param_cvae.eval()
    else:
        if args.load:
            base_search_dir = "./results"
        
            # Hnet 선택 (Pnet는 문자열 치환으로 자동 유추)
            rel_Hnet_path = interactive_file_selector(
                prompt_msg="제안 모델 가중치 (hidden_cvae.pth)를 선택하세요 (param_cvae.pth는 자동 로드됨):", 
                start_dir=base_search_dir
            )
            Hnet_path = os.path.join(base_search_dir, rel_Hnet_path)
            Pnet_path = Hnet_path.replace('hidden_cvae.pth', 'param_cvae.pth')
            if os.path.exists(Hnet_path) and os.path.exists(Pnet_path):
                print(f"\n\033[1;32m[Phase 3] --load flag detected. Loading selected weights...\033[0m")
                print(f"Loading weights from:\n - {Hnet_path}\n - {Pnet_path}")
                
                hidden_cvae.load_state_dict(torch.load(Hnet_path, map_location=config.DEVICE))
                param_cvae.load_state_dict(torch.load(Pnet_path, map_location=config.DEVICE))
                hidden_cvae.eval()
                param_cvae.eval()
            else:
                print(f"\n\033[1;31m[Warning] --load flag provided, but weight files not found. Starting training...\033[0m")
        else:
            print(f"\n[Phase 3] Starting Independent Training...")   
            trainer = ProbabilisticTrainer(train_loader, val_loader, config, hidden_cvae, param_cvae)
            hidden_cvae, param_cvae, history = trainer.train()
    
    # =================================================================
    # Phase 4: Analysis on Simulation Data
    # =================================================================
    run_full_analysis(
        hidden_cvae=hidden_cvae, param_cvae=param_cvae, baseline_model=None,
        test_loader=test_loader, config=config, system=system, normalizer=normalizer
    )

    # =================================================================
    # Phase 5: Real Clinical Data Validation
    # =================================================================
    print("\n\033[1;36m=== Processing Real Clinical Data (Sumner) ===\033[0m")
    try:
        real_loader = RealOGTTDataLoader(file_path='data/clean_sumner_n_612.xlsx', config=config, split_file='data/data_split_indices.json')
        obs_real, hid_real, params_real, _ = real_loader.load_data()
        
        # Real 데이터를 동일한 Normalizer로 정규화
        X_real = normalizer.normalize_inputs(torch.FloatTensor(obs_real).view(len(obs_real), -1).to(config.DEVICE), 'observed')
        Y_real = normalizer.normalize_inputs(torch.FloatTensor(hid_real).view(len(hid_real), -1).to(config.DEVICE), 'hidden')
        P_real = normalizer.normalize_params(torch.FloatTensor(params_real).to(config.DEVICE))
        
        dataset_real = TensorDataset(X_real, Y_real, P_real)
        
        # 테스트 스플릿만 추출
        with open('data/data_split_indices.json', 'r') as f:
            test_indices = json.load(f)['test_indices']
        real_test_set = Subset(dataset_real, test_indices)
        real_test_loader = DataLoader(real_test_set, batch_size=config.BATCH_SIZE, shuffle=False)
        
        # Real Data에 대해서도 Analysis 실행 (폴더 이름을 다르게 저장하도록 config 임시 변경)
        config.EXPERIMENT_NAME += "_REAL"
        run_full_analysis(
            hidden_cvae=hidden_cvae, param_cvae=param_cvae, baseline_model=None,
            test_loader=real_test_loader, config=config, system=system, normalizer=normalizer
        )
        
    except Exception as e:
        import traceback
        print(f"\n[Warning] Real data validation failed or skipped.")
        traceback.print_exc()

if __name__ == "__main__":
    main()