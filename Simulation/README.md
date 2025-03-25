# Simulation Algorithms for Rough Heston Model

Ruirong Feng

This directory contains implied volatility simulation algorithms for the rough Heston model. 

`iv_simu_hybrid_scheme.m` and `iv_simu_markov_approx.m` are implementations for the classical hybrid scheme and the simulation of Markovian approximation process (lifted Heston model), respectively.

We proudly introduce the **Accelerated Hybrid Scheme** as a new simulation algorithm for the rough volatility models. This new idea has reached a great balance between precision and speed. The implementation is detailed in `iv_simu_pc.m`, and the paper is coming soon.

`exact_v3` is a benchmark implementation for the rough Heston model's characteristic function, thanks to Chenyu Wang. 

Mar 25, 2025
