import os
import sys
import copy
import json
import torch
import numpy as np

# 프로젝트 루트 경로 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..')) 
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import Config
from src.data_loader import setup_dataloaders, DataGenerator
from src.models import HiddenVarPredictor, ParameterEstimator, SingleNetworkBaseline
from src.analyzer import get_analyzer_class
from tools.exp_tools import set_seed, get_system_class
from tools.interactive_file_selector import interactive_file_selector

# =====================================================================
# 1. 안전하게 가중치를 로드하는 Helper 함수
# =====================================================================
def load_weight_safe(model, path, device, possible_keys=None):
    if possible_keys is None:
        possible_keys = ['state_dict', 'model_state_dict', 'baseline_state_dict']
        
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict):
        for key in possible_keys:
            if key in ckpt:
                model.load_state_dict(ckpt[key])
                return
        for k, v in ckpt.items():
            if 'state_dict' in k and 'optimizer' not in k:
                model.load_state_dict(v)
                return
    model.load_state_dict(ckpt)

# =====================================================================
# 2. 메인 실행부
# =====================================================================
def main():
    print("\n\033[1;33m=== [논문 Figure 생성을 위한 가중치 파일 선택] ===\033[0m")
    base_search_dir = "./results"
    
    # 1) Config 선택
    rel_config_path = interactive_file_selector("[1/4] 실험 설정 파일 (config.json) 선택:", start_dir=base_search_dir)
    config_path = os.path.join(base_search_dir, rel_config_path)
    
    # 2) Baseline (Single Net) 선택
    rel_baseline_path = interactive_file_selector("[2/4] Baseline (Single Net) 가중치 선택:", start_dir=base_search_dir)
    baseline_path = os.path.join(base_search_dir, rel_baseline_path)
    
    # 3) Ours (With Spectral Norm) 선택
    rel_Hnet_sn_path = interactive_file_selector("[3/4] Ours (WITH Spectral Norm) Hidden Net 선택:", start_dir=base_search_dir)
    ours_h_sn_path = os.path.join(base_search_dir, rel_Hnet_sn_path)
    ours_p_sn_path = ours_h_sn_path.replace('Hnet', 'Pnet')
    
    # 4) Ours (Without Spectral Norm) 선택 [Ablation 용]
    rel_Hnet_no_sn_path = interactive_file_selector("[4/4] Ours (WITHOUT Spectral Norm) Hidden Net 선택:", start_dir=base_search_dir)
    ours_h_no_sn_path = os.path.join(base_search_dir, rel_Hnet_no_sn_path)
    ours_p_no_sn_path = ours_h_no_sn_path.replace('Hnet', 'Pnet')

    # 1. 설정 및 시스템 초기화
    config = Config()
    print(f"\n[Info] Loading Global Config from: {config_path}")
    with open(config_path, 'r') as f:
        for k, v in json.load(f).items():
            if not k.startswith('__'): setattr(config, k, v)
                
    config.SYSTEM_NAME = 'sir'
    set_seed(config.SEED)
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    
    SystemClass = get_system_class(config.SYSTEM_NAME)
    system = SystemClass()

    exp_config = {'USE_SDE': False, 'OOD_SPLIT': True}

    # 2. 데이터 로드
    gen_config = copy.deepcopy(config)
    gen_config.USE_SDE = False
    print("\n[Info] Loading Cache & Setting up DataLoaders...")
    generator = DataGenerator(system, gen_config)
    sim_data_tuple = generator.generate_data() 
    
    loaders = setup_dataloaders(exp_config, sim_data_tuple, system, config)
    _, val_loader, test_loader, _, _, normalizer = loaders

    # 3. 모델 초기화 (3가지 아키텍처 모두 메모리에 로드)
    sample_x, sample_y, sample_p = next(iter(test_loader))
    x_dim, y_dim, p_dim = sample_x.shape[1], sample_y.shape[1], sample_p.shape[1]

    print("[Info] Initializing 3 Models (Baseline, SN, No-SN)...")
    
    # [Model 1] Baseline
    baseline_net = SingleNetworkBaseline(x_dim, p_dim, model_config=config.MODEL_CONFIG['param_net'], use_spectral_norm=None).to(device)
    
    # [Model 2] Ours (WITH Spectral Norm)
    hidden_net_sn = HiddenVarPredictor(x_dim, y_dim, p_dim, model_config=config.MODEL_CONFIG['hidden_net'], use_spectral_norm=True).to(device)
    param_net_sn = ParameterEstimator(x_dim, y_dim, p_dim, model_config=config.MODEL_CONFIG['param_net'], use_spectral_norm=True).to(device)

    # [Model 3] Ours (WITHOUT Spectral Norm)
    hidden_net_no_sn = HiddenVarPredictor(x_dim, y_dim, p_dim, model_config=config.MODEL_CONFIG['hidden_net'], use_spectral_norm=False).to(device)
    param_net_no_sn = ParameterEstimator(x_dim, y_dim, p_dim, model_config=config.MODEL_CONFIG['param_net'], use_spectral_norm=False).to(device)

    # 4. 가중치 로드
    print("[Info] Loading Checkpoints...")
    try:
        load_weight_safe(baseline_net, baseline_path, device)
        load_weight_safe(hidden_net_sn, ours_h_sn_path, device)
        load_weight_safe(param_net_sn, ours_p_sn_path, device)
        load_weight_safe(hidden_net_no_sn, ours_h_no_sn_path, device)
        load_weight_safe(param_net_no_sn, ours_p_no_sn_path, device)
        print("✅ 모든 모델(5개의 .pth)을 성공적으로 로드했습니다.")
    except Exception as e:
        print(f"❌ 체크포인트 로드 실패: {e}")
        return

    baseline_net.eval(); hidden_net_sn.eval(); param_net_sn.eval(); hidden_net_no_sn.eval(); param_net_no_sn.eval()

    # 5. OOD 테스트 데이터 추론 (Inference)
    print("\n🚀 추론을 시작합니다...")
    p_true_list, p_sn_list, p_no_sn_list, p_base_list, x_obs_list = [], [], [], [], []

    with torch.no_grad():
        for test_x, _, test_p in test_loader:
            test_x = test_x.to(device)
            
            # Baseline 예측
            p_base_raw = normalizer.denormalize_params(baseline_net(test_x))
            
            # Ours (SN) 예측
            p_curr_sn = torch.zeros_like(p_base_raw) 
            for _ in range(config.ITERATIONS):
                y_hat = hidden_net_sn(test_x, p_curr_sn)
                p_curr_sn = param_net_sn(test_x, y_hat)
            p_sn_raw = normalizer.denormalize_params(p_curr_sn)
            
            # Ours (No-SN) 예측
            p_curr_no_sn = torch.zeros_like(p_base_raw) 
            for _ in range(config.ITERATIONS):
                y_hat = hidden_net_no_sn(test_x, p_curr_no_sn)
                p_curr_no_sn = param_net_no_sn(test_x, y_hat)
            p_no_sn_raw = normalizer.denormalize_params(p_curr_no_sn)
            
            p_true_list.append(normalizer.denormalize_params(test_p.to(device)).cpu().numpy())
            p_sn_list.append(p_sn_raw.cpu().numpy())
            p_no_sn_list.append(p_no_sn_raw.cpu().numpy())
            p_base_list.append(p_base_raw.cpu().numpy())
            x_obs_list.append(normalizer.denormalize_inputs(test_x, 'observed').cpu().numpy())
            break

    p_true = np.concatenate(p_true_list, axis=0)
    p_sn = np.concatenate(p_sn_list, axis=0)
    p_no_sn = np.concatenate(p_no_sn_list, axis=0)
    p_base = np.concatenate(p_base_list, axis=0)
    x_obs = np.concatenate(x_obs_list, axis=0)

    # 6. Analyzer 호출 및 Figure 2종(Main, Ablation) 분리 추출
    print("\n🎨 Figure 생성을 시작합니다...")
    target_dir = os.path.dirname(ours_h_sn_path)
    AnalyzerClass = get_analyzer_class(config.SYSTEM_NAME)
    
    # 6-1. Main Results (Ours with SN vs Baseline)
    print(" -> 1. Generating Main Results (Spectral Norm ON)")
    analyzer_sn = AnalyzerClass(hidden_net_sn, param_net_sn, normalizer, config, system)
    analyzer_sn.results_path = os.path.join(target_dir, 'figures_Main_SN')
    os.makedirs(analyzer_sn.results_path, exist_ok=True)
    analyzer_sn.evaluate_simulation(p_true=p_true, p_ours=p_sn, p_base=p_base, x_obs=x_obs)
    
    # 6-2. Ablation Results (Ours without SN vs None)
    # 비교군은 필요없으므로 p_base 자리에 zeros 전달
    print(" -> 2. Generating Ablation Results (Spectral Norm OFF)")
    analyzer_no_sn = AnalyzerClass(hidden_net_no_sn, param_net_no_sn, normalizer, config, system)
    analyzer_no_sn.results_path = os.path.join(target_dir, 'figures_Ablation_No_SN')
    os.makedirs(analyzer_no_sn.results_path, exist_ok=True)
    analyzer_no_sn.evaluate_simulation(p_true=p_true, p_ours=p_no_sn, p_base=np.zeros_like(p_base), x_obs=x_obs)
    
    print(f"\n🎉 작업 완료! 2개의 폴더에 그림이 저장되었습니다:")
    print(f" 1. {analyzer_sn.results_path} (메인 결과)")
    print(f" 2. {analyzer_no_sn.results_path} (발산하는 Ablation 결과)")

if __name__ == "__main__":
    main()