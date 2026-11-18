import numpy as np
import torch

"""
Physics System

This file contains information about the Spin System. 

- SpinEnv              :  defines the physical system: spins, couplings, lattice structure, and energy calculations
- build_neighbour_list :  defines the grid’s interacting neighbors
- build_all_pairs      :  defines the SK model’s interacting pairs

"""

def build_neighbour_list(N):
    pairs, seen = [], set()
    for r in range(N):
        for c in range(N):
            i = r * N + c
            for nr, nc in [((r + 1) % N, c), (r, (c + 1) % N)]:
                j = nr * N + nc
                if (i, j) not in seen:
                    pairs.append((i, j))
                    seen.add((i, j))
    return pairs


def build_all_pairs(D):
    """
    Defines the SK model's interacting pairs
    """
    return [(i, j) for i in range(D) for j in range(i + 1, D)]


# =====================================================
#  ENVIRONMENTS
# =====================================================
class SpinEnv:
    """
    Unified +-1 spin environment.

    kind = "ising"  : N x N grid, nearest-neighbor, uniform J=+1
    kind = "ea"     : N x N grid, nearest-neighbor, bimodal J=+-1
    kind = "sk"     : D spins, fully connected, bimodal J=+-1/sqrt(D)

    LEGACY API: root(), step_index(), step(), encode(), state_to_x(),
    energy(), and log_reward() below implement an older tuple-based,
    single-trajectory state representation. See the LEGACY API NOTE at
    the top of this file -- none of the three files currently known to
    import this module call these methods; the actual batched training
    and reward path uses energy_batch() and direct tensor indexing
    instead. Kept here, unmodified, rather than removed.
    """
    def __init__(self, kind, N=None, D=None, p_af=0.5, J_seed=0):
        self.kind = kind
        if kind in ("ising", "ea"):
            assert N is not None
            self.N = N
            self.D = N * N
            self.PAIRS = build_neighbour_list(N)
        elif kind == "sk":
            assert D is not None
            self.N = None
            self.D = D
            self.PAIRS = build_all_pairs(D)
        else:
            raise ValueError(f"unknown kind={kind!r}")

        self.p_af = p_af
        if kind == "ising":
            self.J = {pair: 1.0 for pair in self.PAIRS}
        elif kind == "ea":
            rng = np.random.default_rng(J_seed)
            signs = np.where(rng.random(len(self.PAIRS)) < p_af, -1.0, 1.0)
            self.J = {pair: float(j) for pair, j in zip(self.PAIRS, signs)}
        elif kind == "sk":
            rng = np.random.default_rng(J_seed)
            signs = np.where(rng.random(len(self.PAIRS)) < p_af, -1.0, 1.0)
            vals = signs / np.sqrt(self.D)
            self.J = {pair: float(j) for pair, j in zip(self.PAIRS, vals)}

        self.n_actions = 2
        self.encode_dim = self.D

        # --- precomputed structures for vectorized/batched ops ---
        # (built lazily/cached; kept separate from the tuple-based root()/
        # step() API above so existing single-trajectory code paths, if
        # anything still uses them, are untouched)
        i_idx = np.array([p[0] for p in self.PAIRS], dtype=np.int64)
        j_idx = np.array([p[1] for p in self.PAIRS], dtype=np.int64)
        Jvals = np.array([self.J[p] for p in self.PAIRS], dtype=np.float32)
        self._pair_i_np, self._pair_j_np, self._Jvals_np = i_idx, j_idx, Jvals
        self._pair_tensors_cache = {}  # device -> (i_idx, j_idx, Jvals) tensors

        # neighbor list for O(degree) local-delta-energy MCMC instead of
        # O(D) or O(pairs) full-energy recomputation per proposed flip
        # (the latter is what the original MetropolisHastings did -- a
        # real, easy-to-fix inefficiency, not just a GFlowNet-side issue)
        self.neighbors = {i: [] for i in range(self.D)}
        for (a, b), Jab in self.J.items():
            self.neighbors[a].append((b, Jab))
            self.neighbors[b].append((a, Jab))

    def _pair_tensors(self, device):
        key = str(device)
        if key not in self._pair_tensors_cache:
            self._pair_tensors_cache[key] = (
                torch.as_tensor(self._pair_i_np, device=device),
                torch.as_tensor(self._pair_j_np, device=device),
                torch.as_tensor(self._Jvals_np, device=device),
            )
        return self._pair_tensors_cache[key]

    def energy_batch(self, X):
        """
        Vectorized energy for a BATCH of full configurations.
        X: [B, D] tensor of +-1 values (any device).
        Returns: [B] tensor of energies.
        Replaces the old per-sample Python-loop energy() for anywhere
        that used to call it once per trajectory in a loop -- this is the
        the same fix in spirit as batching the GFlowNet policy calls,
        just applied to the reward computation instead.
        """
        i_idx, j_idx, Jvals = self._pair_tensors(X.device)
        return -(Jvals.unsqueeze(0) * X[:, i_idx] * X[:, j_idx]).sum(dim=1) # No external field, energy is -J \sum_{<i,j>} s_i s_j where <i,j> are neighbors

    def local_delta_energy(self, x, i):
        """
        Energy change from flipping site i alone, using only that site's
        neighbors (O(degree) instead of recomputing the full energy over
        all pairs). x: numpy array of +-1 values. Standard result:
        dE = 2 * x[i] * h_i, where h_i = sum_j J_ij * x[j] over neighbors.
        """
        h_i = sum(Jij * x[j] for j, Jij in self.neighbors[i])
        return 2.0 * x[i] * h_i

    def root(self):
        """LEGACY -- see the LEGACY API NOTE at the top of this file and
        the class docstring above. Appears unused: the batched training
        path never calls this."""
        return (None,) * self.D

    def step_index(self, s):
        """LEGACY -- see root()."""
        for i, v in enumerate(s):
            if v is None:
                return i
        return self.D

    def step(self, s, a):
        """LEGACY -- see root()."""
        i = self.step_index(s)
        s = list(s)
        s[i] = -1 if a == 0 else +1
        return tuple(s)

    def energy(self, x):
        """LEGACY -- see root(). Superseded by energy_batch() (vectorized,
        actually used by the training/reward path) and energy_fast_np()
        (vectorized single-config version, used by the MCMC diagnostics
        loop). This version does a Python dict-lookup per pair and is
        slow if called in a tight loop."""
        x = np.asarray(x, dtype=np.float32)
        return -sum(self.J[(i, j)] * x[i] * x[j] for i, j in self.PAIRS)

    def energy_fast_np(self, x):
        """Vectorized numpy energy for a SINGLE configuration, using the
        same cached pair-index arrays as energy_batch (built once in
        __init__). This is the version to call from a tight loop --
        e.g. once per raw MCMC step for an autocorrelation-time
        diagnostic -- since plain energy() above does a slow per-pair
        Python dict-lookup loop (fine for one-off calls, far too slow
        to call tens of thousands of times per T-point)."""
        x = np.asarray(x, dtype=np.float32)
        return float(-np.sum(self._Jvals_np * x[self._pair_i_np] * x[self._pair_j_np]))

    def log_reward(self, terminal, T):
        """LEGACY -- see root(). Depends on energy() and state_to_x(),
        both themselves flagged legacy/unused; the actual training path
        computes log-reward inline from energy_batch() instead (see
        TBGFlowNet.loss / KCGFlowNet.loss)."""
        return -self.energy(self.state_to_x(terminal)) / T

    def encode(self, state):
        """LEGACY -- see root()."""
        return torch.tensor(
            [0.0 if v is None else float(v) for v in state], dtype=torch.float32)

    def state_to_x(self, state):
        """LEGACY -- see root()."""
        return np.array([float(v) for v in state], dtype=np.float32)
