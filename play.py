"""
Web-based GUI for Alternating Networks Experiment Runner using Gradio.

Installation:
    pip install gradio

Usage:
    python gui.py
    (Then open the provided local URL in your web browser, e.g., http://127.0.0.1:7860)
"""
import os
import glob
import traceback
import json
import gradio as gr
from dataclasses import asdict

# 기존 프로젝트 모듈 임포트
from config import Config
from main import run_experiment_pipeline

def run_experiment(system_name, exp_name, epochs, batch_size, lr, seed, 
                   run_baseline, use_spectral_norm, use_derivative, derivative_method,
                   h_dims_str, h_act, p_dims_str, p_act):
    """
    Gradio 버튼 클릭 시 실행될 콜백 함수.
    사용자 입력을 Config 객체로 변환하고 메인 파이프라인을 실행합니다.
    """
    status_log = ""
    images = []
    
    try:
        # 1. Config 객체 생성 및 기본 사용자 입력 덮어쓰기
        config = Config()
        config.SYSTEM_NAME = system_name
        config.EXPERIMENT_NAME = exp_name
        config.EPOCHS = int(epochs)
        config.BATCH_SIZE = int(batch_size)
        config.LEARNING_RATE = float(lr)
        config.SEED = int(seed)
        config.RUN_BASELINE = run_baseline
        
        # 2. 고급 옵션 및 모델 아키텍처 덮어쓰기
        config.USE_SPECTRAL_NORM = use_spectral_norm
        config.USE_DERIVATIVE = use_derivative
        config.DERIVATIVE_METHOD = derivative_method
        
        def parse_dims(dim_str):
            # "64, 64, 64" 형태의 문자열을 정수 리스트로 변환
            return [int(d.strip()) for d in dim_str.split(',') if d.strip().isdigit()]

        config.MODEL_CONFIG = {
            'hidden_net': {
                'hidden_dims': parse_dims(h_dims_str),
                'activation': h_act
            },
            'param_net': {
                'hidden_dims': parse_dims(p_dims_str),
                'activation': p_act
            }
        }
        
        # 3. 설정 재빌드 (기존 config.py 하위 호환성 보장)
        if hasattr(config, '_build_experiments'):
            config._build_experiments()
        else:
            # 오리지널 config.py를 위한 수동 재빌드 (AttributeError 해결)
            config.EXPERIMENTS = []
            main_loss_name = config.LOSS_CONFIG[-1][0] if config.LOSS_CONFIG else "vanilla"
            for scen in config.SCENARIOS:
                exp = scen.copy()
                exp['NAME'] = f"{scen['NAME']}_{main_loss_name}"
                config.EXPERIMENTS.append(exp)
                
        # 4. JSON 저장 (재현성을 위한 설정 기록)
        target_dir = os.path.join(config.RESULTS_DIR, config.SYSTEM_NAME, config.EXPERIMENT_NAME)
        os.makedirs(target_dir, exist_ok=True)
        
        if hasattr(config, 'save_to_json'):
            config.save_to_json(os.path.join(target_dir, 'experiment_config.json'))
        else:
            with open(os.path.join(target_dir, 'experiment_config.json'), 'w') as f:
                json.dump(asdict(config), f, indent=4)
        
        status_log += f"[*] Starting experiment: {exp_name} on {system_name}...\n"
        status_log += f"  - Spectral Norm: {use_spectral_norm}\n"
        status_log += f"  - Derivative Features: {use_derivative} (Method: {derivative_method})\n"
        if not run_baseline:
            status_log += f"  - H_phi Dims: {config.MODEL_CONFIG['hidden_net']['hidden_dims']} ({h_act})\n"
        status_log += f"  - P_psi Dims: {config.MODEL_CONFIG['param_net']['hidden_dims']} ({p_act})\n\n"
        
        # 5. 파이프라인 실행 (main.py의 함수 호출)
        run_experiment_pipeline(config)
        
        status_log += f"\n[+] Experiment completed successfully!\n"
        status_log += f"[*] Results saved to: {target_dir}\n"
        
        # 6. 생성된 결과 이미지들 수집
        image_paths = glob.glob(os.path.join(target_dir, "*.png"))
        
        if image_paths:
            images = image_paths
            status_log += f"[*] Found {len(image_paths)} result plots."
        else:
            status_log += "\n[!] No plots found. Check if the analyzer ran correctly."
            
    except Exception as e:
        status_log += f"\n[!] Error during execution:\n{str(e)}\n"
        status_log += traceback.format_exc()
        
    return status_log, images

# =====================================================================
# 동적 UI 업데이트 함수
# =====================================================================
def toggle_h_net_visibility(is_baseline):
    """Baseline 체크 여부에 따라 H_phi 네트워크 패널의 표시 여부를 결정합니다."""
    # Baseline이 선택되면 H_phi 패널 숨김 (visible=False)
    return gr.update(visible=not is_baseline)

# =====================================================================
# Gradio UI Layout Definition
# =====================================================================
with gr.Blocks(title="Alternating Networks Dashboard", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🚀 Alternating Networks Experiment Runner")
    gr.Markdown("""
    **Quick Start Guide:**
    * **Tab 1: Global Settings**
      1. **Select** your target dynamical system.
      2. **Set** basic hyperparameters (epochs, batch size, learning rate, seed).
      3. **Configure** advanced data features like derivative methods.
    * **Tab 2: Model Architecture**
      4. **Toggle** Baseline mode or Spectral Normalization.
      5. **Design** your neural network architecture (Hidden layers & Activation).
    * **Run & Monitor**
      6. **Click 'Run Experiment'** and monitor the live logs and evaluation plots on the right.
    """)
    
    with gr.Row():
        # --------------------------------------------------
        # 좌측 패널: 실험 설정 (Inputs)
        # --------------------------------------------------
        with gr.Column(scale=1):
            
            with gr.Tab("1. Global Settings"):
                system_name = gr.Dropdown(
                    choices=['lotka_volterra', 'sir', 'ogtt_simul'], 
                    value='lotka_volterra', 
                    label="Dynamical System"
                )
                exp_name = gr.Textbox(value="gui_experiment", label="Experiment Name")
                
                with gr.Row():
                    epochs = gr.Number(value=1000, label="Epochs", precision=0)
                    batch_size = gr.Number(value=256, label="Batch Size", precision=0)
                
                with gr.Row():
                    lr = gr.Number(value=0.001, label="Learning Rate")
                    seed = gr.Number(value=42, label="Random Seed", precision=0)
                
                gr.Markdown("#### 🔧 Advanced Data Features")
                use_derivative = gr.Checkbox(label="Use Derivative Features (Lagrangian)", value=True)
                
                derivative_method = gr.Dropdown(
                    choices=['finite_diff', 'spline', 'lagrange', 'polynomial'], 
                    value='spline', 
                    label="Derivative Method"
                )

            with gr.Tab("2. Model Architecture"):
                run_baseline = gr.Checkbox(label="Run Baseline (Single Net) Instead", value=False)
                use_spectral_norm = gr.Checkbox(label="Apply Spectral Normalization", value=True)
                
                gr.Markdown("---")
                
                # H_phi 네트워크 설정 (Baseline 클릭 시 숨겨짐)
                with gr.Group() as h_net_group:
                    gr.Markdown("#### 🧠 Hidden-State Predictor")
                    h_dims_str = gr.Textbox(value="64, 64, 64", label="Hidden Layers (comma separated)")
                    h_act = gr.Dropdown(choices=["SiLU", "ReLU", "Tanh", "Sigmoid"], value="SiLU", label="Activation")
                
                # P_psi 네트워크 설정 (Baseline일 경우 이 설정만 사용됨)
                with gr.Group() as p_net_group:
                    gr.Markdown("#### 🧠 Parameter Estimator / Baseline Net")
                    p_dims_str = gr.Textbox(value="64, 64, 64", label="Hidden Layers (comma separated)")
                    p_act = gr.Dropdown(choices=["SiLU", "ReLU", "Tanh", "Sigmoid"], value="SiLU", label="Activation")
            
            run_btn = gr.Button("▶️ Run Experiment", variant="primary")
            
        # --------------------------------------------------
        # 우측 패널: 결과 확인 (Outputs)
        # --------------------------------------------------
        with gr.Column(scale=2):
            gr.Markdown("### 📊 Results & Logs")
            
            # 로그 출력창
            status_output = gr.Textbox(label="Status / Logs", lines=7, interactive=False)
            
            # 생성된 그래프를 갤러리 형태로 보여줌
            gallery_output = gr.Gallery(
                label="Evaluation Plots", 
                show_label=True, 
                elem_id="gallery",
                columns=[2], 
                rows=[2], 
                object_fit="contain", 
                height="auto"
            )
            
    # --- Event Listeners ---
    
    # Baseline 체크박스가 변경될 때 H_phi 패널의 가시성을 동적으로 업데이트
    run_baseline.change(
        fn=toggle_h_net_visibility,
        inputs=[run_baseline],
        outputs=[h_net_group]
    )
    
    # Run 버튼 클릭 시 실험 실행 (새로 추가한 derivative_method가 반드시 inputs에 포함되어야 동작합니다)
    run_btn.click(
        fn=run_experiment,
        inputs=[
            system_name, exp_name, epochs, batch_size, lr, seed, 
            run_baseline, use_spectral_norm, use_derivative, derivative_method,
            h_dims_str, h_act, p_dims_str, p_act
        ],
        outputs=[status_output, gallery_output]
    )

if __name__ == "__main__":
    # 서버 실행 시 로컬 호스트 및 포트 지정
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)