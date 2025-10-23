# How to Run LAN-C

## Prerequisites
- Windows: MinGW-GCC (via MSYS2: pacman -S mingw-w64-x86_64-gcc mingw-w64-x86_64-make)
- Python 3 + numpy, matplotlib, scikit-learn, gzip (for MNIST)

## Step 1: Generate Datasets
cd data
python gen_spirals.py   # → spirals.bin (2in,2out)
python gen_moons.py     # → moons.bin
python gen_mnist.py     # → mnist_train.bin, mnist_test.bin (784in,10out)

## Step 2: Build
mkdir bin
make clean
make all   # → bin/xor.exe etc. (or main_spirals.exe for spirals.bin)

## Step 3: Run Experiments
# XOR (all types; edit main_xor.c acts[] for ablation)
bin/xor.exe  # Logs to experiments/results/xor_<type>_<seed>.csv

# Spirals
bin/spirals.exe

# MNIST (subset)
bin/mnist.exe

# For seeds: Edit srand_seed(seed++) in main; loop via batch script:
for %s in (0 1 2 3 4) do bin/xor.exe %s  # Pseudo; impl loop in C or bat

# Baselines: Set acts=FIXED_RELU in code, rebuild.

## Step 4: Visualize
cd viz
python plot_training.py ../experiments/results/xor_poly_42.csv  # Curves PNG
python plot_acts.py ../experiments/results/xor_poly_42.csv poly  # Evol
python plot_ablation.py  # Heatmap (gen ablations.csv manually: act,seed,acc)

## Ablations
- Edit configs/*.h for arch/acts.
- Run all → aggregate CSVs in Excel/Python for table.

## Troubleshooting
- Compile err: Check #includes.
- Data load fail: Verify .bin header.
- Slow: Subset data; batch=1 OK for toy.

Outputs in experiments/results/: CSVs → viz PNGs.