# Learnable Activation Neural Networks in Pure C (LAN-C)

This repository implements a compact, educational neural-network framework in pure C that supports learnable/parametric activation functions. It focuses on transparency (no external NN libraries), portability, and a small feature set sufficient for experiments and coursework.

The codebase contains training/eval kernels, multiple parametric activation types, utilities to generate toy datasets, Python scripts for visualization, and an ablation runner to sweep combinations of activation choices and initialization strategies.

## Table of contents

- Project overview
- What is included (code layout)
- Supported activations and initialization strategies
- Configuration & tunables
- Build & run (Windows / PowerShell) â€” step-by-step
- Running experiments (XOR, spirals, MNIST)
- Ablation runner (automation)
- Visualization (Python plotting)
- Tests / validation (numeric grad check)
- Troubleshooting & tips
- License

## Project overview

LAN-C is a small research/teaching project that demonstrates how activation functions can be parameterized and learned during training. The project implements forward and backward passes through activation parameters and provides mechanisms to initialize, regularize, and constrain these parameters for numerical stability.

Goals:
- Demonstrate learnable activation functions (PReLU, cubic polynomial, piecewise linear, Swish)
- Implement backpropagation through activation parameters and update them with the optimizer
- Provide reproducible experiments (XOR, spirals, MNIST) and tools for ablation studies

## Code layout

- `src/` â€” C source code and headers (core NN code)
  - `activations.c` / `activations.h` â€” activation implementations, forward/backward, init strategies
  - `layer.c` / `layer.h` â€” dense layer, activation wiring and caches
  - `network.c` / `network.h` â€” network construction, forward, backward, train loop
  - `optimizer.c` / `optimizer.h` â€” SGD updates (weights, biases, activation params), supports momentum and per-parameter act lrs
  - `data.c` / `data.h` â€” dataset loaders / generators
  - `utils.c` / `utils.h` â€” matrix ops, logging, helpers
  - `config.c` / `config.h` â€” central tunables for numeric stability and activation bounds

- `obj/` â€” compiled objects and temporary generated mains created by the ablation runner
- `bin/` â€” optional compiled executables (not required)
- `data/` â€” small data utilities and included MNIST idx files; also contains `gen_*.py` generators
- `experiments/` â€” results and ablation summary CSVs
- `viz/` â€” Python plotting scripts to visualize activation evolution and training curves
- `scripts/` â€” helper scripts (e.g., `run_ablation.py`) to automate sweeps
- `main_xor.c`, `main_spirals.c`, `main_mnist.c` â€” example experiment drivers

## Supported activation functions

The project implements the following activation types; each activation stores a small parameter vector (possibly length 0 for fixed activations) and computes gradients w.r.t. its parameters during backprop:

- PRELU (parametric ReLU): a learned slope for negative inputs (1 parameter per neuron)
- POLY_CUBIC (cubic polynomial): ax^3 + bx^2 + cx + d (4 parameters per neuron)
- PIECEWISE (piecewise linear): a small set of breakpoints and slopes (multiple params per neuron)
- SWISH (parametric swish): x * sigmoid(beta * x) with learnable beta (1 parameter)
- FIXED_RELU, FIXED_SIG (non-learnable baselines)

Note: parameter counts depend on the implementation and whether activations are shared per neuron or per-channel. See `activations.c` for the exact layout.

### Initialization strategies

Activations support multiple initialization strategies (controlled by the `ActInitStrategy` enum used in `init_act` / `init_layer` / `init_net`):

- `ACT_INIT_DEFAULT` â€” sensible default (e.g., PRELU=0.25, POLY_CUBIC = identity-ish)
- `ACT_INIT_RANDOM_SMALL` â€” small random perturbations around zero/default
- `ACT_INIT_NOISY` â€” small Gaussian noise added to params
- `ACT_INIT_IDENTITY` â€” (if available) initialize act to behave like identity

Use these when constructing the network to study sensitivity to initialization.

## Configuration & tunables

Configuration for numerical stability and activation constraints is centralized in `src/config.h` / `src/config.c`.

Key tunables exposed in the code (defaults live in `config.c`):
- `ACT_PARAM_MIN` and `ACT_PARAM_MAX` â€” clamp activation parameters after updates to keep them within safe bounds
- `ACT_Z_CLIP_B` â€” clipping bound used for intermediate `z` computations in some actives (e.g., PIECEWISE) to avoid overflow
- `ACT_GRAD_CLIP_NORM` â€” L2 norm threshold to clip activation-parameter gradients
- `GRAD_CLIP_NORM` â€” L2 norm threshold to clip weight/bias gradients globally

Optimizer-level options (in the `SGD` optimizer struct) include:
- `lr` â€” base learning rate
- `momentum` â€” classical SGD momentum
- `act_lr` or per-activation/per-layer `act_lr` multipliers â€” multiply the base lr for activation parameters
- `act_grad_clip` â€” clip thresholds for activation parameter gradients (overrides global config when set)

To change defaults for experiments, modify `src/config.c` or wire setters (the code includes setter functions).

## Build & run (Windows / PowerShell)

Prerequisites
- A C compiler (recommended: MinGW-w64 GCC on Windows). Make sure `gcc` is on your PATH.
- Python 3 (for data generation, ablation runner, and plotting)
- Python packages for plotting: pandas, matplotlib, seaborn (pip install pandas matplotlib seaborn)

Typical build steps (PowerShell):

1) From project root, try the Makefile (if you have make installed):

```powershell
# from project root
make
```

2) If `make` is not available, compile manually with `gcc` (example for XOR binary):

```powershell
# compile the XOR example (adapt paths as needed)
gcc -I src -std=c99 -O2 src/utils.c src/config.c src/activations.c src/layer.c src/network.c src/data.c src/optimizer.c src/main_xor.c -o obj/xor.exe -lm

# run it
.\obj\xor.exe
```

3) To build other mains, replace `main_xor.c` with `main_spirals.c` or `main_mnist.c` in the compile command and change the output name.

Notes:
- The project uses only the C standard library math (`-lm`). No other C dependencies are required.
- If you see function-signature or missing-symbol errors after modifying code, a `make clean` or removing stale object files under `obj/` then recompiling helps.

## Data generation

Small generators are available in `data/`:

- `data/gen_mnist.py` â€” helper to download or reformat MNIST files (if needed)
- `data/gen_spirals.py`, `data/gen_moons.py`, etc. â€” generate toy datasets used by `main_spirals.c` and friends

Example (PowerShell):

```powershell
# generate spiral dataset (script prints where files are written)
python data/gen_spirals.py
```

The projects ships small MNIST idx files in `data/` (train/test) so you can run the MNIST example without an extra download.

## Running experiments (examples)

1) XOR experiment (small & fast)

```powershell
# compile
gcc -I src -std=c99 -O2 src/utils.c src/config.c src/activations.c src/layer.c src/network.c src/data.c src/optimizer.c src/main_xor.c -o obj/xor.exe -lm
# run
.\obj\xor.exe
# output: per-epoch CSV written to experiments/results/ (see printed path)
```

2) Spirals experiment

```powershell
gcc -I src -std=c99 -O2 src/utils.c src/config.c src/activations.c src/layer.c src/network.c src/data.c src/optimizer.c src/main_spirals.c -o obj/spirals.exe -lm
.\obj\spirals.exe
```

3) MNIST experiment (longer; CPU-only - expect minutes to hours depending on network size)

```powershell
gcc -I src -std=c99 -O2 src/utils.c src/config.c src/activations.c src/layer.c src/network.c src/data.c src/optimizer.c src/main_mnist.c -o obj/mnist.exe -lm
.\obj\mnist.exe
```

Each run writes a CSV into `experiments/results/` with per-epoch metrics (loss, acc) and the activation-parameter values across epochs. The ablation runner also writes `experiments/ablations.csv` with summary metrics per run.

## Ablation runner (automation)

`scripts/run_ablation.py` automates sweeping combinations of datasets, activation choices, and initialization schemes. It:

- generates temporary `main_*.c` files with hard-coded activation choices and init strategies
- compiles each generated main into `obj/` using `gcc`
- runs the binaries and extracts final loss & accuracy
- appends one row per completed run to `experiments/ablations.csv`

How to run:

```powershell
# run the full ablation (may take long). Ensure Python 3 is available.
python scripts/run_ablation.py
```

Notes and resume behavior
- The runner appends to `experiments/ablations.csv`. If it detects a run already logged, it will skip or you can manually prune the CSV to resume. On Windows the script writes relative paths for embedded log filenames to avoid C string escape issues.

## Visualization (Python)

Minimal plotting scripts are in `viz/`:

- `viz/plot_training.py` â€” plot loss/accuracy curves from a run CSV
- `viz/plot_acts.py` â€” show activation parameter evolution across epochs
- `viz/plot_ablation.py` â€” aggregate ablation CSV into heatmaps or bar charts

Example usage (PowerShell):

```powershell
pip install pandas matplotlib seaborn
python viz/plot_training.py experiments/results/spirals_poly_cubic_act_init_default_results_42.csv
python viz/plot_acts.py experiments/results/spirals_poly_cubic_act_init_default_results_42.csv
python viz/plot_ablation.py experiments/ablations.csv
```

If you prefer, open the CSV with Excel/LibreOffice to inspect values directly.

## Tests & validation

- `src/act_grad_check.c` (built as a small binary) performs numeric gradient checks comparing analytic activation-parameter gradients with finite-difference approximations. Running this is the quickest sanity test after changes to activation code.

Build & run the grad check:

```powershell
gcc -I src -std=c99 -O2 src/utils.c src/config.c src/activations.c src/act_grad_check.c -o obj/act_grad_check.exe -lm
.\obj\act_grad_check.exe
```

Expected: printed analytic vs numeric gradients for supported activation types and small differences within numeric tolerance.

## Troubleshooting & tips

- "make" not found on Windows: use the `gcc` commands shown above (MinGW-w64 recommended).
- Compilation errors after code changes: remove stale object files under `obj/` and re-run the compile command.
- C string literal / path problems on Windows: the ablation generator uses relative forward-slash paths to avoid backslash escape issues â€” if you modify the script, ensure paths inserted into generated `.c` files are escaped or use forward slashes.
- Long MNIST runs: reduce network size in `main_mnist.c` or lower training epochs for faster iteration.
- Adding new activations: implement forward, backward (w.r.t inputs and params), an init routine, and ensure `act_reg` / parameter clamping use `src/config` values.

Developer tips

- Prefer explicit field initializers for structs in C when adding fields later to avoid silent initializer ordering bugs.
- Run `act_grad_check` after modifying activation code.
- Keep per-run CSVs (in `experiments/results/`) to enable plotting and reproducible comparisons.

## Contributing

1. Fork and create branches for features/fixes.
2. Run `act_grad_check` and a small XOR/spirals run before making a PR.
3. Include short tests or reproduce steps in the PR description.

## License
MIT

## Contact / References
- Original ideas: He et al. (PReLU) and standard references for Swish and piecewise parametric activations.
