import numpy as np


# =====================================================
#  Observables / susceptibility
# =====================================================
def magnetization_susceptibility(env, samples, T):
    """
    Standard ferromagnetic susceptibility (Ising):
      chi = D * (<m^2> - <|m|>^2) / T
    """
    X = np.asarray(samples, dtype=np.float32)
    m = X.mean(axis=1)
    absm = np.abs(m)
    chi = env.D * (float((m ** 2).mean()) - float(absm.mean()) ** 2) / T
    return chi, float(absm.mean())

def spin_glass_overlap_moments(env, samples, n_pairs=None):
    """
    Two-replica overlap trick: split independent equilibrium samples into
    disjoint pairs (a, b) from the SAME disorder realization, compute
    q_ab = (1/D) sum_i s_i^a s_i^b, and return its raw overlap values
    plus 2nd/4th moments. This needs no ground truth / enumeration --
    only i.i.d. equilibrium samples, which is exactly what both MCMC and
    a trained GFlowNet give.

    Used for both:
      - chi_SG = D * <q^2>                      (spin-glass susceptibility)
      - U4     = 1 - <q^4> / (3 <q^2>^2)         (Binder cumulant)

    chi_SG saturates rather than peaking as T->0 at small D (limited by
    ground-state degeneracy), so it does NOT have a clean turnover point
    to extract T_c from at N=5 -- see binder_cumulant_sg() and the
    crossover_from_binder() estimator in run_tc_sweep.py, which is the
    standard, more robust alternative used in spin-glass simulations.
    """
    X = np.asarray(samples, dtype=np.float32)
    n = len(X)
    if n_pairs is None:
        n_pairs = n // 2
    idx = np.random.permutation(n)
    qs = np.array([float(np.mean(X[idx[2 * k]] * X[idx[2 * k + 1]]))
                   for k in range(n_pairs)])
    q2 = float(np.mean(qs ** 2))
    q4 = float(np.mean(qs ** 4))
    return qs, q2, q4


def spin_glass_susceptibility(env, samples, T, n_pairs=None):
    """chi_SG = D * <q^2>. See spin_glass_overlap_moments() for caveats."""
    qs, q2, q4 = spin_glass_overlap_moments(env, samples, n_pairs)
    chi_sg = env.D * q2
    return chi_sg, float(np.mean(np.abs(qs)))


def binder_cumulant_sg(env, samples, n_pairs=None):
    """
    Binder cumulant of the overlap distribution:
      U4 = 1 - <q^4> / (3 <q^2>^2)
    U4 -> 0 for a Gaussian (paramagnetic, disordered) overlap distribution
    and U4 -> 2/3 for a fully bimodal q=+-1 distribution (frozen, near-
    degenerate ground states). It's dimensionless and monotonic across
    the crossover, so unlike chi_SG it doesn't need a peak to exist --
    the standard indicator for locating a spin-glass transition/crossover
    when true finite-size scaling (multiple N, looking for where curves
    cross) isn't being done.
    """
    qs, q2, q4 = spin_glass_overlap_moments(env, samples, n_pairs)
    U4 = 1.0 - q4 / (3.0 * q2 ** 2 + 1e-12)
    return U4


def susceptibility(env, samples, T):
    if env.kind == "ising":
        return magnetization_susceptibility(env, samples, T)
    else:  # ea, sk
        return spin_glass_susceptibility(env, samples, T)


def order_parameter_samples_for_hist(env, samples):
    """LEGACY -- appears unused. Return a 1D array to histogram: m for
    Ising, pairwise overlap q for EA/SK. make_plots.py's
    cmd_distributions computes the same quantity with its own inline
    code rather than calling this -- its own comment says "pairwise
    overlap via disjoint-pair trick, matching spin_glass_lib", i.e. it
    was written to replicate this function's logic, not call it. Kept
    here, unmodified, rather than removed."""
    if env.kind == "ising":
        X = np.asarray(samples, dtype=np.float32)
        return X.mean(axis=1)
    else:
        qs, _, _ = spin_glass_overlap_moments(env, samples)
        return qs