# Efficient Option Pricing & Implied Volatility Calculation under Rough Heston Model

This repository implements simulation and pricing methods for rough volatility models, focusing on the **Rough Heston Model**. Key contributions include:

- Developed a robust Fourier transform inversion approach using the Adams scheme solver for the fractional Riccati equation, providing a stable benchmark for option pricing under rough volatility.
- Proposed the **Accelerated Hybrid Scheme**, which embeds a precomputation step into the classical hybrid scheme to achieve fast and accurate simulation.
- Introduced a Rough Volatility Expansion technique that extends beyond affine Volterra models, enabling accurate and stable implied volatility surfaces for general two-factor rough dynamics.


## 📖 Overview

- Implements the Accelerated Hybrid Scheme, leveraging Markovian approximation (Abi Jaber, 2018) and fast Fourier transform to accelerate the classical hybrid scheme (Bennedsen et al., 2017).
- Strong convergence and efficient time complexity for simulating stochastic Volterra equations are guaranteed by theoretical analysis. 
- Includes numerical experiments validating the algorithm on the rough Heston model (the case with non-Lipschitz coefficients).

## 🧮 Implemented Methods

In addition to the Accelerated Hybrid Scheme (`accelerated_hybrid.m`) and Fourier transform inversion (`Fourier_Transform_Inversion/`), the following Monte Carlo schemes are available for comparison:

- **Euler–Maruyama discretization** (`euler_scheme.m`)
- **Classical Hybrid Scheme** (`standard_hybrid.m`)
- **Implicit–Explicit Markov Approximation** (`markov_approx.m`)

## 📂 Repository Structure

- `main.m` — Entry point for running simulations.
- `accelerated_hybrid.m`, `euler_scheme.m`, `standard_hybrid.m`, `markov_approx.m` — Core simulation algorithms.
- `convergence_test.m`, `testConvergenceAcchyb.m`, `testConvergenceJ.m`, `testConvergenceN.m`, `testConvergenceNf.m` — Convergence and validation scripts.
- `Fourier_Transform_Inversion/` — Fourier-based pricing and implied volatility calculation.
- `assets/` — Precomputed results, figures, and tables.

## 📋 Requirements

- **MATLAB R2024a** or later
- Financial Toolbox (for implied volatility calculation)
- Symbolic Math Toolbox

## 🚀 Quickstart

1. **Set your MATLAB path** to the project root (so all `.m` files and subfolders are visible).
2. **Open** `main.m` and run:
   ```matlab
   % In the Editor or command window:
   run('main.m')
   ```

## 📈 Results & Figures

- Precomputed results and figures are available in the `assets/` folder.
- Additional figures and tables for convergence and implied volatility are provided.

## 📜 References

1. Abi Jaber, E., Oct. 2018. Stochastic Invariance and Stochastic Volterra Equations. Theses, Université Paris sciences et lettres
2. Bennedsen, M., Lunde, A., Pakkanen, M. S., Jun. 2017. Hybrid scheme for Brownian semistationary processes. Finance and Stochastics 21 (4), 931–965.
