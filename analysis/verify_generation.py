# analysis/verify_generation.py
import sys
import os
import copy
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wasserstein_distance
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from pathlib import Path

try:
    import umap
except ImportError:
    umap = None

# 프로젝트 루트 경로 설정
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from config import Config
from data_loader import DataGenerator, RealOGTTDataLoader
from systems.ogtt_simul import OgttSimul
from utils import euler_maruyama, ExperimentLogger # euler_maruyama 임포트 추가

# 시각화 스타일 설정
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['axes.unicode_minus'] = False

def plot_time_series(t_points, real_data, ode_data, sde_data, var_name, save_path):
    """시계열 분포 비교: Real(Boxplot) vs ODE(Line) vs SDE(Tube)"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 1. Real Data (Boxplot)
    df_real = pd.DataFrame(real_data, columns=t_points)
    df_melt = df_real.melt(var_name='Time', value_name='Value')
    sns.boxplot(x='Time', y='Value', data=df_melt, ax=ax, color='gray', width=0.4, boxprops=dict(alpha=0.4))
    
    # 2. Simulation Data (일부만 시각화)
    n_plot = min(100, len(ode_data))
    
    # SDE (Blue Tube)
    for i in range(n_plot):
        ax.plot(range(len(t_points)), sde_data[i], color='royalblue', alpha=0.05, linewidth=1)
    ax.plot([], [], color='royalblue', label='SDE (Proposed)', linewidth=2) 
        
    # ODE (Red Lines)
    for i in range(n_plot):
        ax.plot(range(len(t_points)), ode_data[i], color='crimson', alpha=0.2, linewidth=1)
    ax.plot([], [], color='crimson', label='ODE (Baseline)', linewidth=2) 

    ax.set_title(f"Time-Series Distribution: {var_name}", fontsize=15)
    ax.set_xlabel("Time (min)")
    ax.set_ylabel(f"{var_name}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    print(f"  -> Saved time-series plot to {save_path}")

def plot_phase_space(real_G, real_I, ode_G, ode_I, sde_G, sde_I, time_idx, t_val, save_path):
    """위상 공간(Phase Space) 분포 비교"""
    plt.figure(figsize=(10, 8))
    
    plt.scatter(sde_G[:, time_idx], sde_I[:, time_idx], c='royalblue', alpha=0.2, s=20, label='SDE', edgecolors='none')
    plt.scatter(ode_G[:, time_idx], ode_I[:, time_idx], c='crimson', alpha=0.6, s=15, label='ODE', marker='x')
    plt.scatter(real_G[:, time_idx], real_I[:, time_idx], c='black', alpha=0.8, s=30, label='Real', marker='*')
    
    plt.title(f"Phase Space Distribution at t={t_val} min", fontsize=15)
    plt.xlabel("Glucose")
    plt.ylabel("Insulin")
    plt.legend()
    
    # 범위 제한 (이상치 제외)
    g_max = np.percentile(real_G[:, time_idx], 99) * 1.2
    i_max = np.percentile(real_I[:, time_idx], 99) * 1.2
    plt.xlim(0, g_max)
    plt.ylim(0, i_max)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  -> Saved phase plot to {save_path}")

def analyze_wasserstein(t_points, real, ode, sde, var_name):
    """Wasserstein Distance 계산 및 비교"""
    w1_ode, w1_sde = [], []
    
    for i in range(len(t_points)):
        w_ode = wasserstein_distance(real[:, i], ode[:, i])
        w_sde = wasserstein_distance(real[:, i], sde[:, i])
        w1_ode.append(w_ode)
        w1_sde.append(w_sde)
        
    return w1_ode, w1_sde

def plot_dimension_reduction(real_flat, ode_flat, sde_flat, method='pca', save_path=None):
    """차원 축소 시각화 (PCA, t-SNE, UMAP)"""
    X = np.vstack([real_flat, ode_flat, sde_flat])
    labels = ['Real'] * len(real_flat) + ['ODE'] * len(ode_flat) + ['SDE'] * len(sde_flat)
    
    print(f"  -> Running {method.upper()} on shape {X.shape}...")
    
    if method == 'pca':
        reducer = PCA(n_components=2)
    elif method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
    elif method == 'umap':
        if umap is None:
            print("  ⚠️ UMAP not installed. Skipping. (pip install umap-learn)")
            return
        reducer = umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=42)
    else:
        raise ValueError(f"Unknown method: {method}")

    X_embedded = reducer.fit_transform(X)
    
    df_emb = pd.DataFrame(X_embedded, columns=['Dim1', 'Dim2'])
    df_emb['Type'] = labels
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df_emb[df_emb['Type']=='SDE'], x='Dim1', y='Dim2', 
                    color='royalblue', alpha=0.15, s=30, label='SDE', edgecolor='none')
    sns.scatterplot(data=df_emb[df_emb['Type']=='ODE'], x='Dim1', y='Dim2', 
                    color='crimson', alpha=0.5, s=20, label='ODE', marker='X')
    sns.scatterplot(data=df_emb[df_emb['Type']=='Real'], x='Dim1', y='Dim2', 
                    color='black', alpha=0.8, s=40, label='Real', marker='*')

    plt.title(f"Dataset Distribution ({method.upper()})", fontsize=15)
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"  -> Saved {method} plot to {save_path}")
    plt.close()

def verify_coverage_sde(num_samples=3):
    """
    Test Set의 환자에 대해 1-to-Many SDE Sampling을 수행하여
    실제 데이터가 SDE 생성 분포(Tube) 내에 들어오는지 검증
    """
    print("\n=== [Step 2] 1-to-Many SDE Coverage Verification (Test Set) ===")
    
    # 1. Test Indices 로드
    split_file = project_root / 'data' / 'data_split_indices.json'
    if not split_file.exists():
        print("Error: Split file not found. Please run create_data_split.py first.")
        return
        
    with open(split_file, 'r') as f:
        split_data = json.load(f)
        test_indices = split_data['test_indices']
    
    print(f"  -> Loaded {len(test_indices)} test indices.")

    # 2. 전체 데이터 로드
    config = Config()
    data_path = project_root / 'data' / 'clean_sumner_n_612.xlsx'
    loader = RealOGTTDataLoader(str(data_path), config)
    X_obs, Y_hid, P_true, t_points = loader.load_data()
    
    # 3. 랜덤 환자 선택
    selected_indices = np.random.choice(test_indices, num_samples, replace=False)
    
    save_dir = current_dir / 'verification_results'
    os.makedirs(save_dir, exist_ok=True)
    
    # 4. 검증 루프
    for idx in selected_indices:
        # 실제 데이터
        y_real_glucose = X_obs[idx, :, 0]
        y_real_insulin = Y_hid[idx, :, 0]
        params_real = P_true[idx] # [si, sigma]
        
        # 시스템 설정
        system = OgttSimul()
        
        # 초기값 세팅 (Steady state 계산 포함)
        g0 = y_real_glucose[0]
        i0 = y_real_insulin[0]
        
        # N5, N6 계산을 위한 임시 모델
        from systems.ogtt_simul import OGTTModel, ode_params, sys_params
        temp_model = OGTTModel(ode_params, sys_params, {'si': params_real[0], 'sigma': params_real[1]})
        n5, n6 = temp_model.find_steady_state_N(g0)
        
        y0 = [g0, i0, n5, n6]
        params_list = [params_real[0], params_real[1]]
        
        # Ensemble 생성
        K = 30
        sim_results_G = []
        
        for _ in range(K):
            # euler_maruyama 호출 (randomness 포함)
            y_sim = euler_maruyama(
                system.drift_func,
                system.diffusion_func,
                system.t_span,
                y0,
                t_points, 
                params_list,
                dt_sim=0.01,
                system=system
            )
            sim_results_G.append(y_sim[0, :]) # Glucose
            
        sim_results_G = np.array(sim_results_G) # (K, T)
        
        # 통계 계산 (90% CI)
        sim_mean = np.mean(sim_results_G, axis=0)
        sim_lower = np.percentile(sim_results_G, 5, axis=0)
        sim_upper = np.percentile(sim_results_G, 95, axis=0)
        
        # Coverage Ratio
        is_covered = (y_real_glucose >= sim_lower) & (y_real_glucose <= sim_upper)
        coverage_ratio = np.mean(is_covered)
        
        # 시각화
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.fill_between(t_points, sim_lower, sim_upper, color='dodgerblue', alpha=0.2, label='SDE 90% CI')
        ax.plot(t_points, sim_mean, color='dodgerblue', linestyle='--', label='SDE Mean')
        ax.plot(t_points, y_real_glucose, 'ro-', label=f'Real Patient {idx}', linewidth=2)
        
        ax.set_title(f"SDE Coverage Check (Patient {idx}) | Coverage: {coverage_ratio*100:.0f}%")
        ax.set_xlabel("Time (min)")
        ax.set_ylabel("Glucose (mg/dL)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        p_save = save_dir / f"coverage_patient_{idx}.png"
        plt.savefig(p_save)
        plt.close(fig)
        print(f"  -> Check Patient {idx}: Coverage={coverage_ratio*100:.1f}% | Saved plot: {p_save.name}")


def verify_generation():
    print("=== Verifying SDE Data Generation & Distribution ===")
    
    # [Clean] 기존 캐시 삭제
    cache_dir = project_root / 'data' / 'ogtt_simul'
    if cache_dir.exists():
        for f in cache_dir.glob("*.npz"):
            try:
                os.remove(f)
            except: pass
        print(f"[Clean] Removed cached files in {cache_dir}")
    
    # 1. 설정 및 데이터 로드
    config = Config()
    config.NUM_SAMPLES = 2000 
    config.USE_LAGRANGIAN = True 
    
    system = OgttSimul()
    
    # A. Real Data
    print("\n[1] Loading Real Data...")
    real_path = project_root / 'data' / 'clean_sumner_n_612.xlsx'
    if not real_path.exists(): real_path = project_root / 'clean_sumner_n_612.xlsx'
    
    real_loader = RealOGTTDataLoader(str(real_path), config)
    X_real, Y_real, _, t_points = real_loader.load_data()
    G_real, I_real = X_real[:, :, 0], Y_real[:, :, 0]
    
    # B. ODE Data
    print("\n[2] Generating ODE Data...")
    config_ode = copy.deepcopy(config)
    config_ode.USE_SDE = False
    gen_ode = DataGenerator(system, config_ode)
    X_ode, Y_ode, _, _ = gen_ode.generate_data()
    G_ode, I_ode = X_ode[:, :, 0], Y_ode[:, :, 0]
    
    # C. SDE Data 
    print("\n[3] Generating SDE Data...")
    config_sde = copy.deepcopy(config)
    config_sde.USE_SDE = True
    config_sde.AUGMENTATION_FACTOR = 30
    gen_sde = DataGenerator(system, config_sde)
    
    X_sde, Y_sde, _, _ = gen_sde.generate_data()
    G_sde, I_sde = X_sde[:, :, 0], Y_sde[:, :, 0]

    save_dir = current_dir / 'verification_results'
    os.makedirs(save_dir, exist_ok=True)

    # --- Step 1: Distribution Matching (기존 기능) ---
    print("\n[Step 1] Global Distribution Matching...")
    plot_time_series(t_points, G_real, G_ode, G_sde, "Glucose", save_dir / 'ts_glucose.png')
    plot_time_series(t_points, I_real, I_ode, I_sde, "Insulin", save_dir / 'ts_insulin.png')
    
    plot_phase_space(G_real, I_real, G_ode, I_ode, G_sde, I_sde, 1, 30, save_dir / 'phase_30min.png')
    plot_phase_space(G_real, I_real, G_ode, I_ode, G_sde, I_sde, 4, 120, save_dir / 'phase_120min.png')

    w1_G_ode, w1_G_sde = analyze_wasserstein(t_points, G_real, G_ode, G_sde, "Glucose")
    w1_I_ode, w1_I_sde = analyze_wasserstein(t_points, I_real, I_ode, I_sde, "Insulin")
    
    print("  -> W1 Distance Summary (Mean):")
    print(f"     Glucose: ODE={np.mean(w1_G_ode):.2f} -> SDE={np.mean(w1_G_sde):.2f}")
    print(f"     Insulin: ODE={np.mean(w1_I_ode):.2f} -> SDE={np.mean(w1_I_sde):.2f}")

    # Dimension Reduction
    real_flat = np.hstack([G_real, I_real])
    ode_flat = np.hstack([G_ode, I_ode])
    sde_flat = np.hstack([G_sde, I_sde])
    idx = np.random.choice(len(real_flat), 500, replace=False) if len(real_flat) > 500 else np.arange(len(real_flat))
    
    plot_dimension_reduction(real_flat[idx], ode_flat[:500], sde_flat[:500], 'pca', save_dir / 'pca_distribution.png')
    plot_dimension_reduction(real_flat[idx], ode_flat[:500], sde_flat[:500], 'tsne', save_dir / 'tsne_distribution.png')
    plot_dimension_reduction(real_flat[idx], ode_flat[:500], sde_flat[:500], 'umap', save_dir / 'umap_distribution.png')
    
    # --- Step 2: Coverage Verification (추가된 기능) ---
    verify_coverage_sde(num_samples=3)
    
    print("\n=== Verification Complete ===")
    print(f"All results saved to: {save_dir}")

if __name__ == "__main__":
    verify_generation()