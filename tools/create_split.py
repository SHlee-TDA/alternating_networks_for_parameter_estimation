# utils/create_split.py
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

def create_split():
    # 1. 프로젝트 경로 설정
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent
    
    data_path = project_root / 'data' / 'clean_sumner_n_612.xlsx'
    if not data_path.exists():
        # 혹시 루트에 있을 경우 대비
        data_path = project_root / 'clean_sumner_n_612.xlsx'
        
    print(f"[Split] Loading data from {data_path}...")
    try:
        df = pd.read_excel(data_path)
    except Exception as e:
        print(f"Error reading excel: {e}")
        return

    # 2. 결측치 제거 (Data Loader와 동일한 로직 적용)
    # 필요한 컬럼 정의
    glu_cols = ['oglu0', 'oglu30', 'oglu60', 'oglu90', 'oglu120']
    ins_cols = ['oins0', 'oins30', 'oins60', 'oins90', 'oins120']
    # 파라미터 컬럼도 결측 여부 확인에 포함 (Ground Truth가 있어야 Test 가능하므로)
    param_cols = ['si', 'sigma'] 

    required_cols = glu_cols + ins_cols + param_cols
    
    # NaN이 하나라도 있는 행 제거
    df_clean = df[required_cols].dropna()
    
    N_clean = len(df_clean)
    print(f"[Split] Original samples: {len(df)}")
    print(f"[Split] Valid samples (No NaN): {N_clean} (Dropped {len(df) - N_clean})")

    if N_clean == 0:
        print("Error: No valid data remaining after dropping NaNs.")
        return

    # 3. Random Split (인덱스 기반)
    # 0부터 N_clean-1 까지의 정수 배열 생성
    valid_indices = np.arange(N_clean)
    
    # 80:20 분할 (Seed 고정)
    train_idx, test_idx = train_test_split(valid_indices, test_size=0.2, random_state=42)
    
    # 4. JSON 저장
    split_info = {
        'source_file': str(data_path.name),
        'total_valid_samples': int(N_clean),
        'train_indices': train_idx.tolist(),
        'test_indices': test_idx.tolist(),
        'random_state': 42,
        'note': "Indices correspond to the numpy arrays returned by RealOGTTDataLoader (NaNs dropped)."
    }
    
    save_dir = project_root / 'data'
    os.makedirs(save_dir, exist_ok=True)
    save_path = save_dir / 'data_split_indices.json'
    
    with open(save_path, 'w') as f:
        json.dump(split_info, f, indent=4)
        
    print(f"[Split] Created split file at: {save_path}")
    print(f" - Train: {len(train_idx)} samples")
    print(f" - Test : {len(test_idx)} samples")

def create_sir_ood_split(data_path, save_path, total_target=10000, test_ratio=0.2, seed=42):
    # Load SIR generated dataset
    data = np.load(data_path)
    params = data['params']  # shape: (N, 2) where columns
    beta, gamma = params[:, 0], params[:, 1]
    R0 = beta / gamma
    
    # Index filtering
    valid_train_pool = np.where(R0 >= 1.2)[0]
    valid_test_pool = np.where(R0 <= 0.8)[0]
    
    # Shuffle
    np.random.seed(seed)
    np.random.shuffle(valid_train_pool)
    np.random.shuffle(valid_test_pool)
    
    # Count targets
    n_test = int(total_target * test_ratio)
    n_val = int(total_target * test_ratio)
    n_train = total_target - n_test - n_val
    
    assert len(valid_train_pool) >= n_train + n_val, "Not enough samples for train/val split"
    assert len(valid_test_pool) >= n_test, "Not enough samples for test split"
    
    # Assign indices
    train_idx = valid_train_pool[:n_train].tolist()
    val_idx = valid_train_pool[n_train:n_train+n_val].tolist()
    test_idx = valid_test_pool[:n_test].tolist()
    
    # Save to JSON
    split_dict = {
        'source_file': str(data_path.name),
        'total_samples': int(total_target),
        'train_indices': train_idx,
        'val_indices': val_idx,
        'test_indices': test_idx,
        'random_state': seed,
        'note': "SIR OOD split based on R0 thresholds (Train/Val: R0>=1.2, Test: R0<=0.8)"
    }

    with open(save_path, 'w') as f:
        json.dump(split_dict, f, indent=4)
        
    print(f"[Split] Created SIR OOD split file at: {save_path}")
if __name__ == "__main__":
    data_path = Path(__file__).parent.parent / 'data' / 'sir' / 'augmented_data_ode_noderiv_20000.npz'
    save_path = Path(__file__).parent.parent / 'data' / 'sir' / 'sir_ood_split.json'
    create_sir_ood_split(data_path, save_path, total_target=10000, test_ratio=0.2, seed=42)
