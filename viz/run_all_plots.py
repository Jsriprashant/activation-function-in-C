"""Run every plotting script in viz/ and collect outputs into viz/results/.

Usage:
    python viz/run_all_plots.py

This runner executes the existing scripts in a safe order and prints progress.
It calls the following scripts (if present):
  - make_plots.py
  - plot_best_by_init.py
  - plot_detailed_by_dataset.py
  - plot_training.py
  - plot_acts.py (skipped unless provided with CSV+act args)
  - plot_ablation.py

Notes:
- Requires Python and the plotting dependencies used by the individual scripts
  (pandas, matplotlib, seaborn, numpy as needed).
- This runner uses subprocess to execute each script in a fresh process and
  streams stdout/stderr to the terminal so you can see progress/errors.
"""
import os
import subprocess
import sys

# By default run only the plotting scripts that work with the repository as-is.
# Some plotting scripts require extra input files or a different CSV schema and
# will be skipped to avoid runtime errors. Use --all to attempt every script.
SAFE_SCRIPTS = [
    'viz/make_plots.py',
    'viz/plot_best_by_init.py',
    'viz/plot_detailed_by_dataset.py',
    'viz/plot_ablation.py'
]

# Scripts that are present but need arguments or a different ablations schema.
SKIPPED_SCRIPTS = [
    'viz/plot_training.py',   # requires a CSV filename argument
    # 'viz/plot_ablation.py' is now supported; keep here commented out if needed
    'viz/plot_acts.py'        # requires csv_path and act_type args
]

# If the user sets this env var we will run plot_acts.py with the provided arg
PLOT_ACTS_ENV = os.environ.get('RUN_PLOT_ACTS')
RUN_ALL_FLAG = '--all' in sys.argv

OUT_DIR = os.path.join('viz', 'results')
os.makedirs(OUT_DIR, exist_ok=True)

def run_script(script, args=None):
    cmd = [sys.executable, script]
    if args:
        cmd += args
    print('\n==> Running:', ' '.join(cmd))
    try:
        res = subprocess.run(cmd, check=False)
        if res.returncode != 0:
            print(f"Script {script} exited with code {res.returncode}")
        else:
            print(f"Script {script} finished successfully")
    except FileNotFoundError:
        print(f"Script not found: {script}")

def main():
    scripts_to_run = SAFE_SCRIPTS if not RUN_ALL_FLAG else (SAFE_SCRIPTS + SKIPPED_SCRIPTS)
    for s in scripts_to_run:
        if os.path.exists(s):
            # If the script is one of the known-skipped ones and we're not in --all, continue
            if (s in SKIPPED_SCRIPTS) and (not RUN_ALL_FLAG):
                print(f"Skipping {s} (requires extra inputs or different CSV schema). Use --all to force.)")
                continue
            run_script(s)
        else:
            print(f"Skipping missing script: {s}")

    # optional: run plot_acts.py if the env var is provided
    if PLOT_ACTS_ENV:
        parts = PLOT_ACTS_ENV.split(':', 1)
        if len(parts) == 2:
            csv_path, act_type = parts
            if os.path.exists(csv_path):
                run_script('viz/plot_acts.py', args=[csv_path, act_type])
            else:
                print(f"plot_acts requested but csv not found: {csv_path}")
        else:
            print('RUN_PLOT_ACTS env var malformed; expected csv_path:act_type')

    print('\nAll plotting scripts attempted. Outputs (if any) are in viz/results/')

if __name__ == '__main__':
    main()
