# analysis/verify_generation.py
import sys
import os
import copy
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
from utils import ExperimentLogger

# 시각화 스타일 설정
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['axes.unicode_minus'] = False

def plot_time_series(t_points, real_data, ode_data, sde_data, var_name, save_path):
    """
    시계열 분포 비교: Real(Boxplot) vs ODE(Line) vs SDE(Tube)
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 1. Real Data (Boxplot)
    df_real = pd.DataFrame(real_data, columns=t_points)
    df_melt = df_real.melt(var_name='Time', value_name='Value')
    sns.boxplot(x='Time', y='Value', data=df_melt, ax=ax, color='gray', width=0.4, boxprops=dict(alpha=0.4))
    
    # 2. Simulation Data
    # 시각화를 위해 일부 샘플만 그림
    n_plot = min(100, len(ode_data))
    
    # SDE (Blue Tube)
    for i in range(n_plot):
        ax.plot(range(len(t_points)), sde_data[i], color='royalblue', alpha=0.05, linewidth=1)
    ax.plot([], [], color='royalblue', label='SDE (Proposed)', linewidth=2) # Legend
        
    # ODE (Red Lines)
    for i in range(n_plot):
        ax.plot(range(len(t_points)), ode_data[i], color='crimson', alpha=0.2, linewidth=1)
    ax.plot([], [], color='crimson', label='ODE (Baseline)', linewidth=2) # Legend

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
    
    # SDE (Background)
    plt.scatter(sde_G[:, time_idx], sde_I[:, time_idx], c='royalblue', alpha=0.2, s=20, label='SDE', edgecolors='none')
    # ODE (Baseline)
    plt.scatter(ode_G[:, time_idx], ode_I[:, time_idx], c='crimson', alpha=0.6, s=15, label='ODE', marker='x')
    # Real (Target)
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
    
    # 데이터 병합
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
        # UMAP 파라미터 (Coverage 확인에 적합한 설정)
        reducer = umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=42)
    else:
        raise ValueError(f"Unknown method: {method}")

    X_embedded = reducer.fit_transform(X)
    
    df_emb = pd.DataFrame(X_embedded, columns=['Dim1', 'Dim2'])
    df_emb['Type'] = labels
    
    # 시각화 (SDE를 맨 밑에 깔고, Real을 맨 위에 그림)
    plt.figure(figsize=(10, 8))
    
    # 1. SDE (Background Cloud)
    sns.scatterplot(data=df_emb[df_emb['Type']=='SDE'], x='Dim1', y='Dim2', 
                    color='royalblue', alpha=0.15, s=30, label='SDE', edgecolor='none')
    
    # 2. ODE (Trajectory Lines/Points)
    sns.scatterplot(data=df_emb[df_emb['Type']=='ODE'], x='Dim1', y='Dim2', 
                    color='crimson', alpha=0.5, s=20, label='ODE', marker='X')
    
    # 3. Real (Target Points)
    sns.scatterplot(data=df_emb[df_emb['Type']=='Real'], x='Dim1', y='Dim2', 
                    color='black', alpha=0.8, s=40, label='Real', marker='*')

    plt.title(f"Dataset Distribution ({method.upper()})", fontsize=15)
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"  -> Saved {method} plot to {save_path}")
    plt.close()


def verify_generation():
    print("=== Verifying SDE Data Generation & Distribution ===")
    
    # [추가] 기존 캐시 파일 삭제 (설정 변경 반영을 위해 필수)
    cache_dir = project_root / 'data' / 'ogtt_simul'
    for f in cache_dir.glob("*.npz"):
        try:
            os.remove(f)
            print(f"[Clean] Removed cached file: {f.name}")
        except:
            pass
    
    # 1. 설정 및 데이터 로드
    config = Config()
    print(f"[DEBUG] Verify Config DIFFUSION_SCALE: {getattr(config, 'DIFFUSION_SCALE', 'Not Set')}")
    config.NUM_SAMPLES = 2000 # 비교를 위해 충분한 수 생성
    config.USE_LAGRANGIAN = True # 미분 포함 (데이터 로더 호환성)
    #config.DIFFUSION_SCALE = 0.1
    
    system = OgttSimul()
    
    # A. Real Data
    print("\n[1] Loading Real Data...")
    real_path = project_root / 'data' / 'clean_sumner_n_612.xlsx '
    if not real_path.exists(): real_path = project_root / 'clean_sumner_n_612.xlsx'
    
    real_loader = RealOGTTDataLoader(str(real_path), config)
    # (N, T, 2) -> 값만 사용 (N, T)
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
    gen_sde = DataGenerator(system, config_sde)
    X_sde, Y_sde, _, _ = gen_sde.generate_data()
    G_sde, I_sde = X_sde[:, :, 0], Y_sde[:, :, 0]

    # 저장 경로
    save_dir = current_dir / 'verification_results'
    os.makedirs(save_dir, exist_ok=True)

    # --- 2. Time-Series Visualization ---
    print("\n[4] Visualizing Time-Series...")
    plot_time_series(t_points, G_real, G_ode, G_sde, "Glucose", save_dir / 'ts_glucose.png')
    plot_time_series(t_points, I_real, I_ode, I_sde, "Insulin", save_dir / 'ts_insulin.png')

    # --- 3. Phase Space Visualization ---
    print("\n[5] Visualizing Phase Space...")
    plot_phase_space(G_real, I_real, G_ode, I_ode, G_sde, I_sde, 1, 30, save_dir / 'phase_30min.png')
    plot_phase_space(G_real, I_real, G_ode, I_ode, G_sde, I_sde, 4, 120, save_dir / 'phase_120min.png')

    # --- 4. Wasserstein Distance ---
    print("\n[6] Calculating Wasserstein Distance...")
    w1_G_ode, w1_G_sde = analyze_wasserstein(t_points, G_real, G_ode, G_sde, "Glucose")
    w1_I_ode, w1_I_sde = analyze_wasserstein(t_points, I_real, I_ode, I_sde, "Insulin")
    
    # 그래프 그리기
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ax[0].plot(t_points, w1_G_ode, 'r-o', label='Real vs ODE')
    ax[0].plot(t_points, w1_G_sde, 'b-s', label='Real vs SDE')
    ax[0].set_title("W1 Distance: Glucose")
    ax[0].legend()
    
    ax[1].plot(t_points, w1_I_ode, 'r-o', label='Real vs ODE')
    ax[1].plot(t_points, w1_I_sde, 'b-s', label='Real vs SDE')
    ax[1].set_title("W1 Distance: Insulin")
    ax[1].legend()
    plt.savefig(save_dir / 'wasserstein_distance.png')
    plt.close()
    
    print("  -> W1 Distance Summary (Mean):")
    print(f"     Glucose: ODE={np.mean(w1_G_ode):.2f} -> SDE={np.mean(w1_G_sde):.2f}")
    print(f"     Insulin: ODE={np.mean(w1_I_ode):.2f} -> SDE={np.mean(w1_I_sde):.2f}")

    # --- 5. Dimension Reduction ---
    print("\n[7] Dimension Reduction (t-SNE)...")
    # Flatten data: (N, T) -> (N, T)
    # Glucose와 Insulin을 합쳐서 (N, 2T) 벡터로 만듦
    real_flat = np.hstack([G_real, I_real])
    ode_flat = np.hstack([G_ode, I_ode])
    sde_flat = np.hstack([G_sde, I_sde])
    
    # 너무 많으면 느리므로 500개씩만 샘플링
    idx = np.random.choice(len(real_flat), 500, replace=False) if len(real_flat) > 500 else np.arange(len(real_flat))
    # ODE/SDE는 샘플 수가 많으므로 500개만 사용
    
    plot_dimension_reduction(real_flat[idx], ode_flat[:500], sde_flat[:500], 'pca', save_dir / 'pca_distribution.png')
    plot_dimension_reduction(real_flat[idx], ode_flat[:500], sde_flat[:500], 'tsne', save_dir / 'tsne_distribution.png')
    plot_dimension_reduction(real_flat[idx], ode_flat[:500], sde_flat[:500], 'umap', save_dir / 'umap_distribution.png')

    
    print("\n=== Verification Complete ===")
    print(f"All results saved to: {save_dir}")

if __name__ == "__main__":
    verify_generation()