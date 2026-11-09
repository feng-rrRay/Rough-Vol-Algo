WEAK VOLTERRA ALGORITHM AND COMPARISONS

Branch: codex/weak-volterra-comparisons
Initialized: 2026-11-08, from main at d341bd6.

This directory brings the current numerical work into the GitHub-linked
repository. The original MATLAB methods remain at the repository root.
The branch contains working research prototypes; it is not a completed
validation of the fast algorithm's weak rate or equal-accuracy speed.

CONTENTS AND STATUS

exact_cell/
  run_rate_test.py and rate_kernels.c: coupled fine/coarse simulations of
  the exact-cell reference, using numerical covariance factorization and
  causal CDQ convolution. Includes independent direct-recursion checks.
  check_references.py: nonlinear two-step conditional-Gaussian quadrature
  and covariance sensitivity checks.
  gaussian_control.py: continuous affine/additive Gaussian control.
  analyze_rate_test.py: analysis of the complete prescribed run matrix.

haar_prototype/
  benchmark_haar_compression.py: energy-weighted Haar sampling and fixed-
  mesh compression comparisons using a direct C solver.
  benchmark_fft_residual_cost.py: CDQ with shared/per-path filters; includes
  independent direct checks. Its timing experiment is not equal-accuracy.
  verify_randomized_haar_correction.py: historical response checks and
  coefficient utilities used by the other prototypes. The response
  correction is an earlier experiment, not an extra step required by the
  present full nonlinear randomized-covariance algorithm.
  Corresponding C sources are included; binaries are built locally.

recorded_results/initial_mesh_test/
  Historical production differences, descriptive fitted orders, metadata,
  and checks imported without modification. These are self-convergence
  diagnostics E[F(Y_2N)-F(Y_N)], not measured continuous weak errors.
  The low-H nonlinear Gaussian-payoff result remains inconclusive.
  validation_original.json refers to the original workspace report; that
  PDF is not included here. Timing fields are not a speed comparison.

experiment_matrix.json: broader planned comparison campaign, including
  which subset has actually been executed.
reference_policy.txt: acceptable reference values and their limitations.
import_manifest.json: source locations and SHA-256 hashes at import.
recorded_results/branch_validation_2026-11-08.json: migration check results.
validate_branch.py: builds the C implementations and checks the migrated
  recursions, coupling, and energy normalization; not a rate experiment.

IMPORTANT DIFFERENCES BEFORE COMBINING EXPERIMENTS

The exact-cell harness uses K(t)=sqrt(2H)*t^(H-1/2), T=1, and six model IDs.
The older Haar prototypes use K(t)=t^(H-1/2)/Gamma(H+1/2), typically x0=0,
q(x)=1+0.8*sin(x), b(x)=-0.2*tanh(x). Do not compare their current numbers
directly. Align kernel normalization, model, maturity and payoff first.

Haar modes are fixed within each path and independently refreshed between
paths. Gaussian channels are independent. The current sampler has an
unbounded cross-path coefficient cache; any measured production storage
claim must account for that cache or replace it with bounded storage.
Its numerical coefficient evaluation and deep-level stability still need
the precision audit in experiment_matrix.json. Existing timing runs use
powers of two; arbitrary-grid CDQ setup has not been validated there.

The deterministic projection implementation, matched factor baselines,
continuous nonlinear reference, and equal-RMSE campaign remain to be built
or validated. Existing Fourier_Transform_Inversion/ provides a separate
rough-Heston option reference candidate, with its own solver/integration
accuracy audit still required. It is not a general-q state-payoff oracle.

SETUP

From this directory, using Python 3.12 and clang on PATH:
  python3.12 -m venv .venv
  .venv/bin/python -m pip install -r requirements.lock.txt
  .venv/bin/python validate_branch.py

The lock records the environment used by the original numerical work.
No virtual environment or compiled library is committed. The migration
was checked with the existing workspace environment, without reinstalling.

SMALL RUN (not a rate-validation claim)

  .venv/bin/python exact_cell/run_rate_test.py --hurst .1 --grids 32 \
    --models 1 3 --power 6 --reps 4 --method mc --tag branch_smoke

REPRODUCE THE PRESCRIBED INITIAL SELF-CONVERGENCE EXPERIMENT

  .venv/bin/python exact_cell/run_rate_test.py --hurst .1 .25 .35 \
    --grids 32 64 128 256 --models 1 3 --power 14 --reps 16 \
    --seed 303000000 --tag production
  .venv/bin/python exact_cell/run_rate_test.py --hurst .1 .35 \
    --grids 128 --models 1 3 --power 12 --reps 16 --seed 404000000 \
    --method mc --tag mc_crosscheck
  .venv/bin/python exact_cell/run_rate_test.py --hurst .1 \
    --grids 512 1024 --models 3 --power 12 --reps 16 --seed 505000000 \
    --method mc --tag low_h_extension
  .venv/bin/python exact_cell/check_references.py
  .venv/bin/python exact_cell/gaussian_control.py
  .venv/bin/python exact_cell/analyze_rate_test.py

Analysis requires the production, cross-check, and extension outputs above.
The extension was exploratory, chosen after seeing the secondary payoff;
it must remain separate from the preselected primary fits.

MIGRATION CHANGES

Source copies retain their original numerical methods. The native Haar
builders and standalone control/check scripts now create output folders
on a clean checkout. The exact-cell C function has an explicit void ctypes
return type. No historical production result was regenerated or altered.
