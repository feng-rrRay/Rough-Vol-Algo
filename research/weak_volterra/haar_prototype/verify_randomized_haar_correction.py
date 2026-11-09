"""Prototype and checks for a randomized Haar quadratic-response correction.

This checks tangent equations, the CDQ implementation, Haar coefficient bounds,
and a two-step analytic quadratic response. It is not a measured fixed-maturity
weak convergence rate. The model has q(x)=1+0.8*sin(x).
"""
import json
import math
from pathlib import Path
import numpy as np
from numpy.polynomial.hermite import hermgauss


def coefficients(x):
    q = 1 + .8*np.sin(x)
    sigma = np.sqrt(q)
    first = .4*np.cos(x)/sigma
    second = -.4*np.sin(x)/sigma - .16*np.cos(x)**2/sigma**3
    return sigma, first, second


def haar_difference(z, alpha):
    """Positive orientation of the normalized Haar integral, without scaling."""
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    small = z < 16
    out[small] = (2*(z[small]-.5)**alpha-z[small]**alpha
                  -(z[small]-1)**alpha)/alpha
    c = z[~small]-.5
    t = .5/c
    # Symmetric binomial expansion avoids cancellation for large lag indices.
    series = np.zeros_like(c)
    binomial = 1.
    for k in range(1, 13):
        binomial *= (alpha-k+1)/k
        if k % 2 == 0:
            series += binomial*t**k
    out[~small] = -2*c**alpha*series/alpha
    return out


def weights(H, n, horizon=1.):
    h, alpha, beta = horizon/n, H+.5, 2*H
    chi = 1/math.gamma(alpha)
    lag = np.arange(1, n+1, dtype=float)
    a = chi*h**H*(lag**alpha-(lag-1)**alpha)/alpha
    v = chi**2*h**beta*(lag**beta-(lag-1)**beta)/beta
    s = np.sqrt(np.maximum(0., v-a*a))
    return np.r_[0., a], np.r_[0., s]


def mode_weights(H, n, level, distance, normalized=True, horizon=1.):
    h, alpha, beta = horizon/n, H+.5, 2*H
    lag = np.arange(n, dtype=float)
    z = lag*(2.**level)+distance
    values = haar_difference(z, alpha)*h**H/math.gamma(alpha)
    if normalized:
        r = 2-beta
        # Stable evaluation of d^(-r)-(d+1)^(-r).
        pd = distance**(-r)*(-math.expm1(-r*math.log1p(1./distance)))
        values /= math.sqrt((-math.expm1(-beta*math.log(2)))*pd)
    else:
        values *= 2.**(-H*level)
    return np.r_[0., values]


def draw_mode(H, n, rng):
    beta = 2*H
    level = int(math.floor(-math.log(rng.uniform())/(beta*math.log(2))))
    cap = int(math.ceil(math.log2(n)))
    if level > cap:
        return np.zeros(n+1), (level, None)
    r = 2-beta
    u = rng.uniform()
    # Avoid forming an arbitrarily large integer for rejected distances.
    if u <= (2**level+1)**(-r):
        return np.zeros(n+1), (level, None)
    distance = int(math.floor(u**(-1/r)))
    assert 1 <= distance <= 2**level
    return mode_weights(H, n, level, distance), (level, distance)


def draw_energy_weighted_mode(H, n, rng):
    """Rejection sampling avoids enumerating the capped Haar covariance.

    Every accepted mode has the exact total residual energy. This preserves
    additive driftless terminal Gaussian marginals. Cost is bounded in
    expectation, with a constant depending on fixed H.
    """
    beta, r = 2*H, 2-2*H
    _, residual = weights(H, n)
    target_energy = float(np.dot(residual, residual))
    first_mode = mode_weights(H, n, 0, 1, normalized=False)
    first_energy = float(np.dot(first_mode, first_mode))
    envelope = first_energy/((-math.expm1(-beta*math.log(2)))
                             *(-math.expm1(-r*math.log(2))))
    attempts = 0
    while True:
        attempts += 1
        proposed, mode = draw_mode(H, n, rng)
        energy = float(np.dot(proposed, proposed))
        assert energy <= envelope*(1+1e-12)
        if energy and rng.uniform() < energy/envelope:
            return proposed*math.sqrt(target_energy/energy), mode, attempts


def source_values(state, xi, eta_plus, eta_mode):
    xp, x0, jp, lp, jw, lw = state
    sp, _, _ = coefficients(xp)
    s0, d0, dd0 = coefficients(x0)
    source = np.zeros((3, 6))
    source[0] = [sp*xi, s0*xi, d0*jp*xi,
                 (dd0*jp*jp+d0*lp)*xi, d0*jw*xi,
                 (dd0*jw*jw+d0*lw)*xi]
    source[1] = [sp*eta_plus, 0., s0*eta_plus,
                 2*d0*jp*eta_plus, 0., 0.]
    source[2] = [0., 0., 0., 0., s0*eta_mode, 2*d0*jw*eta_mode]
    return source


def solve_response(a, s, w, xi, eta_plus, eta_mode, cdq=True):
    """Rows: Xplus, Xcoarse, Jplus, Lplus, Jmode, Lmode."""
    n = len(xi)
    kernels = np.array([a, s, w])
    state = np.zeros((6, n+1))
    sources = np.zeros((3, 6, n+1))

    def leaf(i):
        if i < n:
            sources[:, :, i] = source_values(state[:, i], xi[i],
                                             eta_plus[i], eta_mode[i])

    def recurse(left, right):
        if right-left == 1:
            leaf(left)
            return
        mid = (left+right)//2
        recurse(left, mid)
        length = right-left
        size = 1 << ((mid-left+length-2).bit_length())
        fs = np.fft.rfft(sources[:, :, left:mid], n=size, axis=-1)
        fk = np.fft.rfft(kernels[:, :length], n=size, axis=-1)
        convolution = np.fft.irfft(np.sum(fs*fk[:, None, :], axis=0),
                                   n=size, axis=-1)
        state[:, mid:right] += convolution[:, mid-left:right-left]
        recurse(mid, right)

    if cdq:
        recurse(0, n+1)
    else:
        for i in range(n):
            leaf(i)
            state[:, i+1:] += np.sum(sources[:, :, i, None]
                                     *kernels[:, None, 1:n+1-i], axis=0)
    return state


def scalar_scheme(a, w, xi, eta, theta):
    n = len(xi)
    x = np.zeros(n+1)
    for i in range(n):
        sig = coefficients(x[i])[0]
        x[i+1:] += sig*(a[1:n+1-i]*xi[i]+theta*w[1:n+1-i]*eta[i])
    return x


def corrected_cosine_sample(H, n, rng):
    a, s = weights(H, n)
    w, mode = draw_mode(H, n, rng)
    xi, ep, ew = rng.normal(size=(3, n))
    z = solve_response(a, s, w, xi, ep, ew)[:, -1]
    xp, x0, jp, lp, jw, lw = z
    first, second = -.7*math.sin(.7*x0), -.49*math.cos(.7*x0)
    up = .5*(second*jp*jp+first*lp)
    uw = .5*(second*jw*jw+first*lw)
    return math.cos(.7*xp)-up+uw, -up+uw, mode


def solve_antithetic_paths(a, s, w, xi, eta_plus, eta_mode,
                           drift_strength=0., h=1., cdq=True):
    """Four full nonlinear paths; no coefficient or payoff derivatives used.

    Rows are original +/- residual and randomized-mode +/- residual.
    Optional smooth nonlinear drift is -drift_strength*tanh(x).
    """
    n = len(xi)
    kernels = np.array([a, s, w, math.sqrt(h)*a])
    state = np.zeros((4, n+1))
    sources = np.zeros((4, 4, n+1))

    def leaf(i):
        if i == n:
            return
        sig = np.sqrt(1+.8*np.sin(state[:, i]))
        sources[0, :, i] = sig*xi[i]
        sources[1, :2, i] = sig[:2]*eta_plus[i]*np.array([1., -1.])
        sources[2, 2:, i] = sig[2:]*eta_mode[i]*np.array([1., -1.])
        sources[3, :, i] = -drift_strength*np.tanh(state[:, i])

    def recurse(left, right):
        if right-left == 1:
            leaf(left)
            return
        mid = (left+right)//2
        recurse(left, mid)
        length = right-left
        size = 1 << ((mid-left+length-2).bit_length())
        fs = np.fft.rfft(sources[:, :, left:mid], n=size, axis=-1)
        fk = np.fft.rfft(kernels[:, :length], n=size, axis=-1)
        convolution = np.fft.irfft(np.sum(fs*fk[:, None, :], axis=0),
                                   n=size, axis=-1)
        state[:, mid:right] += convolution[:, mid-left:right-left]
        recurse(mid, right)

    if cdq:
        recurse(0, n+1)
    else:
        for i in range(n):
            leaf(i)
            state[:, i+1:] += np.sum(sources[:, :, i, None]
                                     *kernels[:, None, 1:n+1-i], axis=0)
    return state


def derivative_free_sample(H, n, rng, drift_strength=0., energy_weighted=False):
    a, s = weights(H, n)
    if energy_weighted:
        w, mode, _ = draw_energy_weighted_mode(H, n, rng)
    else:
        w, mode = draw_mode(H, n, rng)
    xi, ep, ew = rng.normal(size=(3, n))
    paths = solve_antithetic_paths(a, s, w, xi, ep, ew,
                                  drift_strength=drift_strength, h=1./n)
    payoffs = np.cos(.7*paths[:, -1])
    correction = .5*(payoffs[2]+payoffs[3]-payoffs[0]-payoffs[1])
    return payoffs[0]+correction, correction, mode


def run():
    rng = np.random.RandomState(20261104)
    H, n = .1, 32
    a, s = weights(H, n)
    w = mode_weights(H, n, 3, 2)
    xi, ep, ew = rng.normal(size=(3, n))
    cdq = solve_response(a, s, w, xi, ep, ew, True)
    direct = solve_response(a, s, w, xi, ep, ew, False)
    cdq_error = float(np.max(np.abs(cdq-direct)))
    assert cdq_error < 2e-12
    eps = 1e-3
    xp = scalar_scheme(a, w, xi, ew, eps)
    xm = scalar_scheme(a, w, xi, ew, -eps)
    x0 = scalar_scheme(a, w, xi, ew, 0.)
    first_error = float(np.max(np.abs((xp-xm)/(2*eps)-direct[4])))
    second_error = float(np.max(np.abs((xp+xm-2*x0)/eps**2-direct[5])))
    assert first_error < 2e-6 and second_error < 2e-6

    path_cdq = solve_antithetic_paths(a, s, w, xi, ep, ew,
                                     drift_strength=.2, h=1./n)
    path_direct = solve_antithetic_paths(a, s, w, xi, ep, ew,
                                        drift_strength=.2, h=1./n, cdq=False)
    path_error = float(np.max(np.abs(path_cdq-path_direct)))
    assert path_error < 2e-12
    driftless_paths = solve_antithetic_paths(a, s, w, xi, ep, ew)
    assert np.max(np.abs(driftless_paths[0]-direct[0])) < 2e-12
    rejected_paths = solve_antithetic_paths(a, s, np.zeros_like(w), xi, ep, ew)
    # A rejected mode is the coarse path, never a zero payoff.
    assert np.max(np.abs(rejected_paths[2]-direct[1])) < 2e-12

    # Conditional Gaussian integration checks E[U_w] for a nonpolynomial q.
    # Exact two-step formula: E X2(theta)^3 = 3*eta*v(theta)*C(theta)*exp(-v/2).
    a2, _ = weights(H, 2)
    w2 = mode_weights(H, 2, 2, 1, normalized=False)
    nodes, quadrature_weights = hermgauss(64)
    z = math.sqrt(2)*nodes
    qw = quadrature_weights/math.sqrt(math.pi)
    xi0, xi1 = z[:, None], z[None, :]
    x1 = a2[1]*xi0
    sig, dsig, ddsig = coefficients(x1)
    x2 = a2[2]*xi0+sig*a2[1]*xi1
    c0 = dsig*w2[1]*a2[1]*xi1+w2[2]
    c1 = sig*w2[1]
    ej2 = c0*c0+c1*c1
    el = ddsig*w2[1]**2*a2[1]*xi1
    conditional_u = 3*x2*ej2+1.5*x2*x2*el
    quadrature_response = float(np.sum(qw[:, None]*qw[None, :]*conditional_u))
    v0, cross0 = a2[1]**2, a2[1]*a2[2]
    exact_response = 3*.8*math.exp(-v0/2)*(w2[1]**2*cross0*(1-v0/2)
                                               +v0*w2[1]*w2[2])
    response_error = abs(quadrature_response-exact_response)
    assert response_error < 2e-12

    # Exercise random rejection and the finite cap; this is a smoke check only.
    samples = [corrected_cosine_sample(H, 64, rng) for _ in range(64)]
    assert all(math.isfinite(x[0]) and math.isfinite(x[1]) for x in samples)
    free_samples = [derivative_free_sample(H, 64, rng, .2) for _ in range(64)]
    assert all(math.isfinite(x[0]) and math.isfinite(x[1]) for x in free_samples)
    # Exact energy matching is the additive terminal-law preservation check.
    _, residual = weights(H, 32)
    target_energy = float(np.dot(residual, residual))
    energy_modes = [draw_energy_weighted_mode(H, 32, rng) for _ in range(32)]
    energy_error = max(abs(float(np.dot(mode, mode))-target_energy)
                       for mode, _, _ in energy_modes)
    assert energy_error < 2e-12
    # Small-grid enumeration checks the covariance of the accepted-mode law.
    small_n = 8
    raw = [mode_weights(H, small_n, j, d, normalized=False)[1:]
           for j in range(int(math.ceil(math.log2(small_n)))+1)
           for d in range(1, 2**j+1)]
    energies = np.array([np.dot(v, v) for v in raw])
    _, small_residual = weights(H, small_n)
    exact_energy = float(np.dot(small_residual, small_residual))
    capped_cov = sum(np.outer(v, v) for v in raw)
    mixture_cov = sum((e/energies.sum())*np.outer(v, v)*(exact_energy/e)
                      for v, e in zip(raw, energies))
    covariance_error = float(np.max(np.abs(mixture_cov
                                   -capped_cov*(exact_energy/energies.sum()))))
    assert covariance_error < 2e-12
    energy_samples = [derivative_free_sample(H, 64, rng, .2, True) for _ in range(16)]
    assert all(math.isfinite(x[0]) and math.isfinite(x[1]) for x in energy_samples)
    report = dict(H=H, cdq_direct_max_error=cdq_error,
                  derivative_free_cdq_direct_max_error=path_error,
                  first_tangent_finite_difference_error=first_error,
                  second_tangent_finite_difference_error=second_error,
                  haar_quadratic_response_quadrature_error=response_error,
                  random_samples_checked=len(samples),
                  derivative_free_samples_checked=len(free_samples),
                  energy_weighted_modes_checked=len(energy_modes),
                  energy_weighted_full_samples_checked=len(energy_samples),
                  energy_matching_max_error=energy_error,
                  energy_weighted_covariance_identity_error=covariance_error,
                  mean_rejection_attempts=float(np.mean([v[2] for v in energy_modes])),
                  accepted_modes=sum(mode[1] is not None for _, _, mode in samples),
                  scope='Implementation and fixed-grid identities; no measured global weak rate.')
    path = Path(__file__).parent/'results'/'randomized_haar_correction_checks.json'
    path.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    run()
