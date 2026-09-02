"""G0 training driver.

Follows the sanctioned per-run configuration mechanism: main.py instantiates
config.Config() with its file defaults (only --system/--run_baseline/--epochs/
--results_dir are CLI-overridable), so to set SEED / NUM_SAMPLES / ITERATIONS etc.
per run we patch config.py's default values in place, invoke `python main.py`, and
restore the original config.py afterwards.

Trains the iterative estimator on the linear_oracle system for each seed in
g0_settings.SEEDS. Metrics are computed separately by evaluate_g0.py.
"""
import os
import re
import sys
import subprocess
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, HERE)

import g0_settings as S  # noqa: E402


CONFIG_PATH = os.path.join(ROOT, 'config.py')
LOG_DIR = os.path.join(HERE, 'logs')


def patch_config(original_text, overrides):
    """Return config.py text with the given `FIELD: type = value` defaults replaced."""
    text = original_text
    for name, value in overrides.items():
        pattern = re.compile(r'(?m)^(\s*' + re.escape(name) + r'\s*:\s*[^=\n]*=\s*).*$')
        new_text, n = pattern.subn(lambda m: m.group(1) + value, text)
        if n != 1:
            raise RuntimeError(f"Expected exactly one match for field '{name}', got {n}")
        text = new_text
    return text


def enable_recurrent_loss(text):
    """Enable the supervised+recurrent composite loss (LOSS_CONFIG is a multi-line field)."""
    pat = re.compile(r'LOSS_CONFIG:[^\n]*=\s*field\(default_factory=lambda:\s*\[.*?\]\s*\)', re.DOTALL)
    repl = ("LOSS_CONFIG: List[Tuple[str, float]] = field(default_factory=lambda: "
            "[('supervised', 1.0), ('recurrent', 1.0)])")
    new_text, n = pat.subn(repl, text)
    if n != 1:
        raise RuntimeError(f"Expected exactly one LOSS_CONFIG match, got {n}")
    return new_text


def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(CONFIG_PATH, 'r') as f:
        original = f.read()

    try:
        for seed in S.SEEDS:
            overrides = {
                'SYSTEM_NAME': f"'{S.SYSTEM_NAME}'",
                'EXPERIMENT_NAME': f"'{S.experiment_name(seed)}'",
                'SEED': str(seed),
                'RESULTS_DIR': f"'{S.RESULTS_DIR}'",
                'RUN_BASELINE': 'False',
                'NUM_SAMPLES': str(S.NUM_SAMPLES),
                'AUGMENTATION_FACTOR': '0',
                'BATCH_SIZE': str(S.BATCH_SIZE),
                'EPOCHS': str(S.EPOCHS),
                'LEARNING_RATE': str(S.LEARNING_RATE),
                'USE_EARLY_STOPPING': 'False',
                'USE_DERIVATIVE': 'False',
                'ITERATIONS': str(S.ITERATIONS),
                'RECURRENT_ITER': str(S.RECURRENT_ITER),
                'USE_SPECTRAL_NORM': str(S.USE_SPECTRAL_NORM),
            }
            patched = enable_recurrent_loss(patch_config(original, overrides))
            with open(CONFIG_PATH, 'w') as f:
                f.write(patched)

            log_path = os.path.join(LOG_DIR, f'train_seed{seed}.log')
            print(f"\n{'='*70}\n[G0] Training seed {seed} -> {S.results_path(seed)}\n"
                  f"     log: {log_path}\n{'='*70}", flush=True)

            with open(log_path, 'w') as log:
                log.write(f"# G0 seed {seed} started {datetime.now().isoformat()}\n")
                log.flush()
                proc = subprocess.run(
                    [sys.executable, 'main.py'],
                    cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                )
            if proc.returncode != 0:
                print(f"[G0] seed {seed} FAILED (return code {proc.returncode}); "
                      f"see {log_path}", flush=True)
                sys.exit(proc.returncode)
            print(f"[G0] seed {seed} done.", flush=True)
    finally:
        with open(CONFIG_PATH, 'w') as f:
            f.write(original)
        print("[G0] Restored original config.py.", flush=True)


if __name__ == '__main__':
    main()
