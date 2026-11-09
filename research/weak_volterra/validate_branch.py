"""Quick migration checks; does not validate a convergence-rate claim."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'

import hashlib
import json
import platform
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE / 'exact_cell'), str(HERE / 'haar_prototype')]
import run_rate_test as exact
import benchmark_haar_compression as haar
import benchmark_fft_residual_cost as fft
from verify_randomized_haar_correction import weights


def main():
    report = {'scope': 'Migration, recursions, coupling and normalization only.',
              'python': sys.version, 'platform': platform.platform(),
              'numpy': np.__version__}
    report['exact_cell'] = exact.validate(exact.native())
    haar.validate(haar.native())
    report['haar_direct'] = 'passed independent path and additive-control checks'
    lib = fft.native()
    report['haar_cdq'] = fft.validate(lib)

    rng = np.random.RandomState(2026110801)
    H, n, paths, count = .1, 32, 7, 4
    sampler = haar.EnergySampler(H, n, cap=5)
    filters = sampler.draw(paths, count, rng)
    energy_error = float(np.max(abs(np.sum(filters**2, axis=2)
                                    - sampler.target / count)))
    assert energy_error < 1e-12
    a, _ = weights(H, n)
    a = a[1:]
    d = a / np.sqrt(n)
    xi = rng.normal(size=(paths, n))
    eta = rng.normal(size=(paths, count, n))
    cdq = fft.GenericCDQ(lib, a, d, filters, leaf=4).solve(xi, eta)
    direct = fft.python_direct(a, d, filters, xi, eta)
    error = float(np.max(abs(cdq-direct)))
    assert error < 3e-12
    report['sampled_haar'] = {'mode_energy_max_error': energy_error,
                              'cdq_vs_python_max_error': error,
                              'fresh_modes_per_path': count,
                              'note': 'Not a distribution, tail or rate certification.'}
    report['source_hashes'] = {
        str(p.relative_to(HERE)): hashlib.sha256(p.read_bytes()).hexdigest()
        for folder in ('exact_cell', 'haar_prototype')
        for p in sorted((HERE / folder).iterdir())
        if p.suffix in ('.py', '.c')
    }
    (HERE / 'branch_validation.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
