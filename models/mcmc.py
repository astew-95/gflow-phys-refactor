import numpy as np


# Default Hyperparams, overwritten by CLI
LR_LOGZ_MULT = 50.0
MCMC_WARMUP  = 50_000     # Run MCMC_WARMUP number of update steps before sampling with MCMC, let system settle
MCMC_THIN    = 50         # save MCMC config every MCMC_THIN updates 

def integrated_autocorr_time(x, c=5.0):
    """
    Integrated autocorrelation time of a 1D time series x, via Sokal's
    automatic-windowing estimator: tau_int(M) = 1 + 2*sum_{t=1}^{M} rho(t),
    where rho is the normalized autocorrelation function, and M is chosen
    as the smallest window where M >= c * tau_int(M) (c=5 is a standard
    default). This is an approximate, standard diagnostic -- not a
    rigorous bound -- used here to flag whether an MCMC chain's samples
    are effectively decorrelated given the configured thinning, not as a
    precise scientific measurement in its own right.

    Returns tau_int (float, >= 1.0). If x is too short or constant
    (autocorrelation undefined), returns float('nan').
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n < 10:
        return float("nan")
    x = x - x.mean()
    var = np.dot(x, x) / n
    if var < 1e-12:
        return float("nan")  # constant series (e.g., chain never moved)

    # autocorrelation via FFT (much faster than direct summation for long chains)
    size = 1
    while size < 2 * n:
        size *= 2
    fx = np.fft.rfft(x, n=size)
    acf = np.fft.irfft(fx * np.conjugate(fx), n=size)[:n].real
    acf /= (var * n)  # normalize so acf[0] == 1

    tau = 1.0
    for M in range(1, n // 2):
        tau = 1.0 + 2.0 * np.sum(acf[1:M + 1])
        if M >= c * tau:
            return max(tau, 1.0)
    return max(tau, 1.0)  # window never satisfied c*tau <= M; best available estimate



# =====================================================
#  MCMC baseline (single chain, burn-in + thinning)
# =====================================================
class MetropolisHastings:
    """
    Single-spin-flip Metropolis, now using the O(degree) local-delta
    energy instead of recomputing the FULL energy (O(pairs)) on every
    single proposed flip. For a D=100 grid this is a ~50x reduction in
    work per step (4 neighbors vs ~200 pairs), on top of just being the
    textbook-correct way to implement Metropolis. This alone makes the
    N=10 MCMC baseline dramatically more affordable, independent of
    anything on the GFlowNet side.
    """
    def __init__(self, env, T):
        self.env = env
        self.T = T
        self.x = np.random.choice([-1., 1.], size=env.D).astype(np.float32)
        self.n_accepted = 0
        self.n_total = 0

    def step(self):
        i = np.random.randint(self.env.D)
        dE = self.env.local_delta_energy(self.x, i)
        self.n_total += 1
        if dE <= 0 or np.random.random() < np.exp(-dE / self.T):
            self.x[i] *= -1
            self.n_accepted += 1

    def run_chain(self, warmup, n_samples, thin, track_energy=False):
        """track_energy: if True, records the FULL-PRECISION single-spin
        energy at every raw step (not just thinned samples) during the
        post-warmup sampling phase, for the autocorrelation-time
        diagnostic. Returns (samples, energy_trace) if track_energy else
        just samples (energy_trace is None in that case)."""
        for _ in range(warmup):
            self.step()
        samples = []
        energy_trace = [] if track_energy else None
        for _ in range(n_samples):
            for _ in range(thin):
                self.step()
                if track_energy:
                    energy_trace.append(self.env.energy_fast_np(self.x))
            samples.append(self.x.copy())
        return (samples, energy_trace) if track_energy else samples

    @property
    def acceptance_rate(self):
        return self.n_accepted / self.n_total if self.n_total > 0 else float("nan")


class WolffCluster:
    """
    Wolff single-cluster algorithm, for uniform-coupling ferromagnetic
    Ising ONLY (env.kind == "ising"). Grows a cluster of aligned spins
    probabilistically and flips it as a block, which sidesteps the
    critical slowing down that single-spin Metropolis suffers from near
    T_c -- exactly the concern raised about using Metropolis as the
    "ground truth" reference for validating an N=10 Ising sampler near
    its (finite-size-shifted) transition. NOT valid for EA/SK: the
    bond-dependent acceptance probability this relies on assumes a
    single uniform coupling strength, which frustrated/disordered models
    don't have.
    """
    def __init__(self, env, T):
        assert env.kind == "ising", "Wolff cluster is only implemented for uniform-J Ising"
        self.env = env
        self.T = T
        self.J = 1.0  # uniform coupling, matches IsingEnv/SpinEnv("ising")
        self.p_add = 1.0 - np.exp(-2.0 * self.J / T)
        self.x = np.random.choice([-1., 1.], size=env.D).astype(np.float32)
        self.n_flips = 0  # cluster flips are always "accepted" by construction,
        # so we track cluster SIZE distribution instead of an accept/reject rate
        self.cluster_sizes = []

    def step(self):
        D = self.env.D
        seed = np.random.randint(D)
        cluster = {seed}
        frontier = [seed]
        s0 = self.x[seed]
        while frontier:
            i = frontier.pop()
            for j, _Jij in self.env.neighbors[i]:
                if j not in cluster and self.x[j] == s0 and np.random.random() < self.p_add:
                    cluster.add(j)
                    frontier.append(j)
        for i in cluster:
            self.x[i] *= -1
        self.n_flips += 1
        self.cluster_sizes.append(len(cluster))

    def run_chain(self, warmup, n_samples, thin, track_energy=False):
        for _ in range(warmup):
            self.step()
        samples = []
        energy_trace = [] if track_energy else None
        for _ in range(n_samples):
            for _ in range(thin):
                self.step()
                if track_energy:
                    energy_trace.append(self.env.energy_fast_np(self.x))
            samples.append(self.x.copy())
        return (samples, energy_trace) if track_energy else samples

    @property
    def acceptance_rate(self):
        """Wolff cluster moves are always accepted (never rejected) by
        construction, so 'acceptance rate' isn't a meaningful diagnostic
        here the way it is for Metropolis -- mean fractional cluster size
        (cluster size / D) is reported instead, in run_mcmc's diagnostics."""
        return float("nan")

    @property
    def mean_cluster_fraction(self):
        return float(np.mean(self.cluster_sizes)) / self.env.D if self.cluster_sizes else float("nan")


def run_mcmc(env, T, n_eval, warmup=50_000, thin=50, method="auto",
             return_diagnostics=False):
    """
    method: "auto" uses Wolff for Ising (recommended -- avoids critical
    slowing down near T_c) and Metropolis for EA/SK (Wolff isn't valid
    for disordered/frustrated couplings). "metropolis" or "wolff" force
    a specific sampler; forcing "wolff" on a non-Ising env raises.
    Wolff's warmup/thin are already in units of whole-cluster flips, not
    single-spin flips, so far fewer of them are needed for equivalent
    decorrelation -- if you force "wolff", consider much smaller
    --mcmc_warmup/--mcmc_thin than the single-spin defaults.

    return_diagnostics: if True, returns (samples, diagnostics) where
    diagnostics is a dict with:
      - 'acceptance_rate': fraction of proposed single-spin flips
        accepted (Metropolis only; NaN for Wolff, which always accepts
        by construction -- see 'mean_cluster_fraction' instead)
      - 'mean_cluster_fraction': mean cluster size / D per flip (Wolff
        only; NaN for Metropolis)
      - 'tau_int': integrated autocorrelation time of the single-spin-
        chain energy trace during the sampling phase (see
        integrated_autocorr_time docstring) -- large tau_int relative
        to --mcmc_thin is a sign the chain hasn't decorrelated between
        recorded samples (i.e. --mcmc_thin is too small, or the chain
        is stuck/mixing poorly, e.g. from basin-trapping at low T).
    """
    if method == "auto":
        method = "wolff" if env.kind == "ising" else "metropolis"
    cls = {"metropolis": MetropolisHastings, "wolff": WolffCluster}[method]
    chain = cls(env, T)

    if not return_diagnostics:
        return chain.run_chain(warmup, n_eval, thin)

    samples, energy_trace = chain.run_chain(warmup, n_eval, thin, track_energy=True)
    diagnostics = {
        "acceptance_rate": chain.acceptance_rate,
        "mean_cluster_fraction": (chain.mean_cluster_fraction
                                   if isinstance(chain, WolffCluster) else float("nan")),
        "tau_int": integrated_autocorr_time(energy_trace),
        "method": method,
    }
    return samples, diagnostics


