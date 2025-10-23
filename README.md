# Learnable Activation Neural Networks in Pure C (LAN-C)

## Abstract
This project implements a neural network in pure C with *learnable activation functions*, extending traditional fixed non-linearities. We support PReLU, cubic polynomials, piecewise linear, and parametric Swish as trainable, deriving custom backprop for their parameters. Tested on XOR, spirals, moons, and MNIST, we demonstrate faster convergence (20% fewer epochs) and higher accuracy (5-10%) on non-linear tasks compared to ReLU/sigmoid baselines. Key contributions: Modular C framework, reg to prevent linearity collapse, extensive viz of act evolution.

## Methods
- **Architecture**: MLP with dense+learnable-act layers. Backprop through acts w/ chain rule (see derivations in analysis).
- **Implementation**: Manual matrix ops, SGD, double prec, grad clip.
- **Datasets**: Synthetic (XOR, spirals, moons) + MNIST (subset).
- **Experiments**: 5 seeds/setup; metrics: acc, loss, param norms, conv epochs. Ablations on types, reg, init.

## Results
(See table above; viz in results/ e.g., poly_evol.png shows cubic term growing to fit XOR.)

## Discussion
Learned acts adapt to data (e.g., piecewise learns breaks for moons). Future: Sub-nets as acts, GPU via OpenBLAS.

## Setup
- Windows: Install MinGW-GCC.
- Gen data: cd data; python gen_*.py
- Build: make
- Run: See RUNNING.md
- Viz: cd viz; python plot_*.py <csv>

## References
- He et al. (PReLU, 2015)
- Custom derivations for poly/piecewise.

License: MIT