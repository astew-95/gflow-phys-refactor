"""
Exact finite-lattice partition function of the 2D Ising ferromagnet on an
L x L PERIODIC lattice (Kaufman 1949), matching SpinEnv("ising", N=L):
E = -sum_<ij> s_i s_j over the 2 L^2 nearest-neighbour bonds, J = 1.
"""
import math
import numpy as np


def _log_terms(L, K):
    k = np.arange(2 * L)
    c = math.cosh(2 * K) / math.tanh(2 * K) - np.cos(np.pi * k / L)
    g = np.arccosh(c)
    g[0] = 2 * K + math.log(math.tanh(K))
    odd, even = g[1::2], g[0::2]
    out = []  # (sign, log|Zi|)
    for gs, fn in ((odd, "cosh"), (odd, "sinh"), (even, "cosh"), (even, "sinh")):
        x = L * gs / 2
        if fn == "cosh":
            out.append((1.0, float(np.sum(np.logaddexp(x, -x)))))   # log(2 cosh x)
        else:
            sgn = float(np.prod(np.sign(x)))
            ax = np.abs(x)
            out.append((sgn, float(np.sum(ax + np.log1p(-np.exp(-2 * ax))))))  # log|2 sinh x|
    return out


def log_Z(L, T):
    K = 1.0 / T
    terms = _log_terms(L, K)
    mx = max(t[1] for t in terms)
    s = sum(sg * math.exp(lt - mx) for sg, lt in terms)
    return math.log(0.5) + 0.5 * L * L * math.log(2 * math.sinh(2 * K)) + mx + math.log(s)


def mean_energy(L, T, h=1e-5):
    """<E> = -d log Z / dK (J = 1)."""
    K = 1.0 / T
    f = lambda k: log_Z(L, 1.0 / k)
    return -(f(K + h) - f(K - h)) / (2 * h)


def specific_heat(L, T, h=1e-4):
    """Total heat capacity C = K^2 d^2 log Z / dK^2 (divide by L^2 for per-spin)."""
    K = 1.0 / T
    f = lambda k: log_Z(L, 1.0 / k)
    return K * K * (f(K + h) - 2 * f(K) + f(K - h)) / (h * h)


if __name__ == "__main__":
    import itertools, torch
    from archive.spin_glass_lib import SpinEnv
    torch.set_default_dtype(torch.float64)
    L = 4
    env = SpinEnv("ising", N=L)
    X = torch.tensor(list(itertools.product([-1.0, 1.0], repeat=L * L)))
    E = env.energy_batch(X).numpy()
    worst = 0.0
    for T in (1.0, 1.5, 2.0, 2.269, 2.5, 3.0, 5.0):
        lz_b = float(np.logaddexp.reduce(-E / T))
        p = np.exp(-E / T - lz_b)
        e_b = float((p * E).sum()); c_b = float((p * E ** 2).sum() - e_b ** 2) / T ** 2
        lz, e, c = log_Z(L, T), mean_energy(L, T), specific_heat(L, T)
        worst = max(worst, abs(lz - lz_b), abs(e - e_b) / abs(e_b), abs(c - c_b) / c_b)
        print(f"T={T:5.3f}  logZ {lz:.10f} vs {lz_b:.10f}   <E> {e:.6f} vs {e_b:.6f}   "
              f"C {c:.5f} vs {c_b:.5f}")
    print(f"worst abs/rel error: {worst:.2e}  ->  {'PASS' if worst < 1e-5 else 'FAIL'}")
    print(f"L=10: logZ(T=2.269) = {log_Z(10, 2.269):.6f}, <E>/N = {mean_energy(10, 2.269) / 100:.6f}")
