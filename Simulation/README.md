# Rough Heston Model Simulation & Implied Volatility Calculation

This repository implements the methods from the paper **“Turbocharging Hybrid Scheme for Rough Volatility Model Simulation”**.  

**Abstract:**  
> We explore the fast and accurate Monte Carlo simulation of the stochastic Volterra equations,  
> with the goal to accelerate option pricing and implied volatility calculation in rough volatility  
> models. Leveraging the Markovian approximation process briefed in Abi Jaber (2018) as the  
> precomputation step, we present the Accelerated Hybrid Scheme, a novel simulation algorithm  
> that uses the fast Fourier transform to accelerate the classical hybrid scheme of Bennedsen et al.  
> (2017). Under globally Lipschitz coefficients with linear growth, this algorithm achieves a strong  
> convergence result and a time complexity of O(n·(nf + log n)) for n time steps and nf factors that  
> approximates the Laplace transform of the kernel. We validate our algorithm through numerical  
> experiments on the rough Heston model, demonstrating its effectiveness even with non-Lipschitz  
> coefficients. In contrast with existing simulation methods, the Accelerated Hybrid Scheme pro-  
> vides a fast solution for rough volatility models with more numerical stability.  
> **Keywords:** rough volatility models, stochastic Volterra equations, Monte Carlo simulation,  
> option pricing, implied volatility.

In addition to our **Accelerated Hybrid Scheme** and the **Fourier transform inversion**, we implement three other Monte Carlo schemes for comparison:

- **Euler–Maruyama discretization** (`euler_scheme.m`)  
- **Classical Hybrid Scheme** by Bennedsen et al. (2017) (`standard_hybrid.m`)  
- **Implicit–Explicit scheme of Markovian approximation** of Abi Jaber (2018) (`markov_approx.m`)

---

## 📋 Requirements

- **MATLAB R2024a** or later  
- Financial Toolbox (for implied volatility calculation)
- Symbolic Math Toolbox

---

## 🚀 Quickstart

1. **Set your MATLAB path** to the project root (so all `.m` files and subfolders are visible).
2. **Open** `main.m`.  
   ```matlab
   % In the Editor or command window:
   run('main.m')
