"""
Offline unit tests verifying mathematical identities, bounds, and accounting
consistency across Theorems 1-4 (zero external data or network required).
"""
import math
import numpy as np
from scipy.stats import norm


def scarf_cantelli_bound(mu: float, sd: float, L: float = None) -> float:
    unbounded = 0.5 * (math.sqrt(mu**2 + sd**2) - mu)
    if L is not None and L < 0 and mu > 0 and (L**2 <= mu**2 + sd**2):
        lower_bnd = (-L * sd**2) / ((mu - L)**2 + sd**2)
        return min(unbounded, lower_bnd)
    return unbounded


def test_theorem1_abel_ttd_identity():
    """Verify exact discrete Abel summation-by-parts identity matches Lean Theorem 1."""
    T = 10
    beta = np.exp(-0.08 * np.arange(T))
    m = np.linspace(5000.0, 12000.0, T)
    K = np.linspace(55000.0, 35000.0, T)
    S = np.array([1.0, 0.96, 0.91, 0.85, 0.78, 0.70, 0.60, 0.48, 0.34, 0.18, 0.0])
    S_star = np.array([1.0, 0.97, 0.93, 0.88, 0.82, 0.75, 0.66, 0.54, 0.40, 0.22, 0.0])
    d = S[:-1] - S[1:]
    d_star = S_star[:-1] - S_star[1:]

    lhs = np.sum(beta * ((S_star[:-1] - S[:-1]) * m + (d_star - d) * K))
    dS = S_star - S
    rhs = sum(dS[t] * (beta[t] * m[t] - (beta[t - 1] * K[t - 1] - beta[t] * K[t])) for t in range(1, T))
    assert math.isclose(lhs, rhs, rel_tol=1e-12, abs_tol=1e-9)


def test_theorem1_churn_pincer_witness():
    """Verify Lean 4 constructive private-to-Medicare offloading witness (5, 40, 50, 1, 2)."""
    c0, A, M, t1, t2 = 5.0, 40.0, 50.0, 1.0, 2.0
    f = lambda h: c0 - A * math.exp(-h * t1) + M * math.exp(-h * t2)
    h_star = math.log((M * t2) / (A * t1)) / (t2 - t1)
    assert f(0.0) == 15.0 and f(0.0) > 0.0
    assert math.isclose(f(h_star), -3.0, rel_tol=1e-12)
    assert f(math.log(8.0) + 0.1) > 0.0
    # Verify interior global optimality on dense grid
    grid = np.linspace(0.0, 5.0, 5000)
    assert np.all(np.array([f(h) for h in grid]) >= f(h_star) - 1e-12)


def test_theorem2_scarf_cantelli_sharpness_and_regime():
    """Verify unconstrained and lower-bounded Scarf-Cantelli bounds and L^2 <= mu^2 + sd^2 regime."""
    rng = np.random.default_rng(20261121)
    for mu, sd in [(100.0, 250.0), (500.0, 300.0), (1.0, 1.0)]:
        # Random empirical draws must satisfy ceiling
        x = rng.normal(mu, sd, size=200000)
        emp_mu, emp_sd = float(np.mean(x)), float(np.std(x))
        emp_sav = float(-np.mean(np.minimum(x, 0.0)))
        assert emp_sav <= scarf_cantelli_bound(emp_mu, emp_sd) + 1e-6

        # Two-point Scarf attainment witness on {-r, +r}
        r = math.sqrt(mu**2 + sd**2)
        p = (r - mu) / (2.0 * r)
        assert math.isclose(p * r, scarf_cantelli_bound(mu, sd), rel_tol=1e-12)

    # Verify Lean 4 catch: at mu=1, sd=1, L=-10, L^2 > mu^2 + sd^2 so unconstrained witness beats lower-bounded formula
    mu, sd, L = 1.0, 1.0, -10.0
    assert L**2 > mu**2 + sd**2
    r = math.sqrt(mu**2 + sd**2)
    assert -r >= L
    unconstrained_val = (r - mu) / 2.0
    wrong_formula_val = (-L * sd**2) / ((mu - L)**2 + sd**2)
    assert unconstrained_val > wrong_formula_val
    assert math.isclose(scarf_cantelli_bound(mu, sd, L), unconstrained_val, rel_tol=1e-12)


def test_theorem2_frechet_submodular_coupling():
    """Verify comonotonic coupling reduces achievable oracle savings."""
    c0, s1, s2, m1, m2 = 500.0, 200.0, 900.0, 100.0, 700.0
    countermono_min = min(c0 - s1 + m2, 0.0) + min(c0 - s2 + m1, 0.0)
    comono_min = min(c0 - s1 + m1, 0.0) + min(c0 - s2 + m2, 0.0)
    assert countermono_min <= comono_min  # more negative => larger savings under countermonotonicity


def test_theorem3_mills_ratio_sign_flip_sandwich():
    """Verify exact Gaussian plug-in sign-flip sandwich mu^2/(mu^2+sd^2) < rho* < mu^2/(mu^2+(2/pi)sd^2)."""
    for mu, sd in [(200.0, 400.0), (500.0, 300.0), (1000.0, 250.0)]:
        gamma = mu / sd
        dc = lambda rho: mu * norm.sf(gamma * math.sqrt(rho)) - sd * math.sqrt(rho) * norm.pdf(gamma * math.sqrt(rho))
        rho_lo = (mu**2) / (mu**2 + sd**2)
        rho_hi = (mu**2) / (mu**2 + (2.0 / math.pi) * sd**2)
        assert dc(rho_lo) > 0.0
        assert dc(rho_hi) < 0.0
