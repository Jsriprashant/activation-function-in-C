"""Remove generated/temp files created by the ablation runner and plotting.

This script deletes:
- obj/main_*.c (generated temporary mains)
- obj/*_act_init_*.exe (generated per-run executables)
- viz/results/*.png (generated plots)
- experiments/results/*.csv (generated run CSVs)
- experiments/ablations.csv
- bin/ directory (if empty)

Run from project root:
    python scripts/clean_generated.py
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OBJ = ROOT / 'obj'
VIZR = ROOT / 'viz' / 'results'
RES = ROOT / 'experiments' / 'results'
ABLA = ROOT / 'experiments' / 'ablations.csv'
BIN = ROOT / 'bin'

def rm_glob(pat):
    for p in ROOT.glob(pat):
        try:
            p.unlink()
            print('Removed', p)
        except Exception as e:
            print('Failed to remove', p, e)

# Remove generated mains
rm_glob('obj/main_*.c')
# Remove generated per-run exes
rm_glob('obj/*_act_init_*.exe')
# Remove generated plot PNGs
if VIZR.exists():
    for p in VIZR.glob('*.png'):
        try:
            p.unlink()
            print('Removed', p)
        except Exception as e:
            print('Failed to remove', p, e)
# Remove generated experiment CSVs (keep .gitkeep if present)
if RES.exists():
    for p in RES.glob('*.csv'):
        if p.name == '.gitkeep':
            continue
        try:
            p.unlink()
            print('Removed', p)
        except Exception as e:
            print('Failed to remove', p, e)
# Remove ablations.csv
if ABLA.exists():
    try:
        ABLA.unlink()
        print('Removed', ABLA)
    except Exception as e:
        print('Failed to remove', ABLA, e)
# Remove bin dir if empty
if BIN.exists():
    try:
        # remove only if empty
        if not any(BIN.iterdir()):
            BIN.rmdir()
            print('Removed empty bin/ directory')
        else:
            print('bin/ not empty; skipping removal')
    except Exception as e:
        print('Failed to remove bin/', e)

print('Cleanup complete')
