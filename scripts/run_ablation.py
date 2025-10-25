#!/usr/bin/env python3
"""
Run ablation experiments by generating temporary main files with different
activation types and initialization strategies, compiling and running them,
then collecting final loss/accuracy into experiments/ablations.csv.

This script does not modify the original main_*.c files. It writes a temp
copy for each configuration, compiles it with gcc, runs the produced exe,
and reads the produced CSV log (the temp main is patched so the log filename
is unique per run).

Usage: python scripts/run_ablation.py

Requires gcc to be available on PATH and python3.
"""
import os
import re
import shutil
import subprocess
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
OBJ = ROOT / 'obj'
RESULTS = ROOT / 'experiments' / 'results'
SCRIPTS = ROOT / 'scripts'

os.makedirs(OBJ, exist_ok=True)
os.makedirs(RESULTS, exist_ok=True)

# Configurable lists
datasets = [
    ('xor', SRC / 'main_xor.c'),
    ('spirals', SRC / 'main_spirals.c'),
    # MNIST is heavy; include but beware of runtime (kept here for completeness)
    ('mnist', SRC / 'main_mnist.c'),
]

act_choices = ['POLY_CUBIC', 'PRELU', 'SWISH', 'PIECEWISE', 'FIXED_RELU']
strat_choices = ['ACT_INIT_DEFAULT', 'ACT_INIT_RANDOM_SMALL', 'ACT_INIT_NOISY']

# Short helper: read arch array from main file to determine number of layers
def parse_arch(src_text):
    m = re.search(r"int\s+arch\s*\[]\s*=\s*\{([^}]*)\}", src_text)
    if not m:
        return None
    nums = [int(x.strip()) for x in m.group(1).split(',') if x.strip()]
    return nums

def patch_main_for_config(src_path, act_choice, strat_choice, out_log_name):
    text = src_path.read_text()
    arch = parse_arch(text)
    if not arch:
        raise RuntimeError(f"Could not parse arch from {src_path}")
    n_layers = len(arch) - 1
    # Build acts array string: hidden layers = act_choice, last layer = FIXED_SIG
    acts = [act_choice] * (n_layers - 1) + ['FIXED_SIG']
    acts_str = ', '.join(acts)
    acts_decl = f"ActType acts[] = {{{acts_str}}};"
    # Replace existing acts[] line
    text = re.sub(r"ActType\s+acts\s*\[\s*\]\s*=\s*\{[^}]*\};", acts_decl, text)

    # Build act_strats array: use strat_choice for hidden layers, IDENTITY for output
    strats = [strat_choice] * (n_layers - 1) + ['ACT_INIT_IDENTITY']
    strats_str = ', '.join(strats)
    strat_decl = f"ActInitStrategy act_strats[] = {{{strats_str}}};"
    # If file already has act_strats declaration, replace; otherwise insert after acts decl
    if re.search(r"ActInitStrategy\s+act_strats\s*\[", text):
        text = re.sub(r"ActInitStrategy\s+act_strats\s*\[\s*\]\s*=\s*\{[^}]*\};", strat_decl, text)
    else:
        # insert after Acts declaration
        text = text.replace(acts_decl, acts_decl + '\n    ' + strat_decl)

    # Replace the log filename sprintf to use out_log_name (use callback to avoid
    # issues with backslashes in Windows paths being interpreted as escapes)
    def _repl(m):
        return f'sprintf(logf, "{out_log_name}", 42);'

    text = re.sub(r'sprintf\(logf,\s*"experiments/results/[^"]*",\s*\d+\);', _repl, text)

    return text

def compile_main(temp_path, out_exe):
    # Build compile command similar to earlier invocations
    srcs = ['utils.c', 'config.c', 'activations.c', 'layer.c', 'network.c', 'data.c', 'optimizer.c']
    srcs = [str(SRC / s) for s in srcs] + [str(temp_path)]
    cmd = ['gcc', '-I', str(SRC), '-std=c99', '-O2'] + srcs + ['-o', str(out_exe), '-lm']
    print('Compiling:', ' '.join(cmd))
    subprocess.run(cmd, check=True)

def run_exe(exe_path):
    print('Running:', exe_path)
    subprocess.run([str(exe_path)], check=True)

def read_final_metrics(log_csv):
    p = Path(log_csv)
    if not p.exists():
        return None
    with p.open() as f:
        rows = list(csv.reader(f))
    if len(rows) < 2:
        return None
    header = rows[0]
    last = rows[-1]
    # header: epoch,loss,acc,...
    idx_loss = header.index('loss')
    idx_acc = header.index('acc')
    return float(last[idx_loss]), float(last[idx_acc])

def append_ablation_row(out_csv, row):
    exists = Path(out_csv).exists()
    with open(out_csv, 'a', newline='') as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(['dataset', 'act_hidden', 'act_init', 'seed', 'final_loss', 'final_acc', 'logfile'])
        w.writerow(row)

def main():
    out_csv = ROOT / 'experiments' / 'ablations.csv'
    seed = 42
    # Keep runs small for tests (limit combinations)
    for ds_name, main_path in datasets:
        for act in act_choices:
            for strat in strat_choices:
                exp_name = f"{ds_name}_{act.lower()}_{strat.lower()}"
                temp_main = OBJ / f"main_{exp_name}.c"
                exe = OBJ / f"{exp_name}.exe"
                rel_log = f"experiments/results/{exp_name}_results_{seed}.csv"
                logfile = ROOT / rel_log
                # patch main (pass relative path into C sprintf to avoid backslash escapes)
                print(f"Preparing {exp_name} for dataset {ds_name}")
                patched = patch_main_for_config(main_path, act, strat, rel_log)
                temp_main.write_text(patched)
                # compile
                try:
                    compile_main(temp_main, exe)
                except subprocess.CalledProcessError:
                    print('Compile failed for', exp_name)
                    continue
                # run
                try:
                    run_exe(exe)
                except subprocess.CalledProcessError:
                    print('Run failed for', exp_name)
                    continue
                # read metrics
                metrics = read_final_metrics(logfile)
                if metrics:
                    loss, acc = metrics
                    append_ablation_row(out_csv, [ds_name, act, strat, seed, loss, acc, str(logfile)])
                    print(f"Logged: {ds_name},{act},{strat},{seed},{loss:.4f},{acc:.4f}")
                else:
                    print('No metrics found for', exp_name)

if __name__ == '__main__':
    main()
