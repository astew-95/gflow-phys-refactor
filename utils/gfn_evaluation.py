# utils/gfn_evaluation.py
import math
import torch
from physics.ising_transfer import logZ_transfer
from utils.save_run_history import preserve_rng


def ising_reference(env, T):
    """Transfer-matrix reference for uniform-J, periodic 5x5 Ising."""
    if (
        env.kind != "ising"
        or env.N != 5
        or any(float(j) != 1.0 for j in env.J.values())
    ):
        raise ValueError("Reference requires uniform J=+1, 5x5 periodic Ising")

    beta = 1 / T
    h = 1e-5
    energy = -(
        logZ_transfer(env.N, beta + h)
        - logZ_transfer(env.N, beta - h)
    ) / (2 * h * env.D)

    # Validation trajectories only; never enumerate all 2^25 states.
    codes = torch.arange(64, dtype=torch.long)
    acts = ((codes[:, None] >> torch.arange(env.D)) & 1).long()
    acts = torch.cat([acts, 1 - acts], 0)

    return dict(
        acts=acts,
        logZ=logZ_transfer(env.N, beta),
        target_observables=dict(energy_per_spin=energy),
        transfer_N=env.N,
    )


def exact_reference(env, T, device="cpu"):
    """Enumerate the normalized target distribution for a small system."""
    codes = torch.arange(1 << env.D, dtype=torch.long, device=device)
    acts = (
        (codes[:, None] >> torch.arange(env.D, device=device)) & 1
    ).long()
    spins = (2 * acts - 1).to(torch.get_default_dtype())

    E = env.energy_batch(spins).double()
    lp = -E / T
    logZ = torch.logsumexp(lp, 0)
    lp = lp - logZ
    p = lp.exp()
    m = spins.double().mean(1)

    return dict(
        acts=acts,
        E=E,
        logp=lp,
        p=p,
        m=m,
        logZ=float(logZ),
    )


def exact_observables(q, ref, T, D):
    """Observables under the enumerated probability vector q."""
    E, m = ref["E"], ref["m"]
    abs_m = (q * m.abs()).sum()
    m2 = (q * m.square()).sum()

    return dict(
        energy_per_spin=float((q * E).sum() / D),
        abs_m=float(abs_m),
        chi=float(D * (m2 - abs_m.square()) / T),
        frac_m_pos=float(q[m > 0].sum()),
        U4=float(
            1 - (q * m.pow(4)).sum() / (3 * m2.square() + 1e-300)
        ),
    )


def evaluate_exact(g, ref, beta, initial_parameters):
    """Evaluate every terminal configuration without Monte Carlo sampling."""
    with preserve_rng(), torch.no_grad():
        raw_lp, entropy, _ = g._replay_batch(ref["acts"])
        raw_lp = raw_lp.double()

        mass = float(raw_lp.exp().sum())
        if not math.isfinite(mass) or abs(mass - 1) > 2e-4:
            raise FloatingPointError(
                f"Terminal probabilities do not sum to one: {mass}"
            )

        lq = raw_lp - torch.logsumexp(raw_lp, 0)
        q = lq.exp()
        p = ref["p"]
        lp = ref["logp"]

        obs = exact_observables(q, ref, g.T, g.env.D)
        target = exact_observables(p, ref, g.T, g.env.D)

        training_logp = -beta * ref["E"] / g.T
        training_logp -= torch.logsumexp(training_logp, 0)

        obs.update(
            kl_qp=float((q * (lq - lp)).sum()),
            kl_pq=float((p * (lp - lq)).sum()),
            tv=float((q - p).abs().sum() / 2),
            training_target_kl_qp=float(
                (q * (lq - training_logp)).sum()
            ),
            probability_mass_before_normalization=mass,
            mean_action_entropy=float((q * entropy.double()).sum()),
            energy_error_per_spin=(
                obs["energy_per_spin"] - target["energy_per_spin"]
            ),
            abs_m_error=obs["abs_m"] - target["abs_m"],
            chi_error=obs["chi"] - target["chi"],
            mode_imbalance=abs(obs["frac_m_pos"] - 0.5),
            parameter_displacement=float(
                torch.sqrt(
                    sum(
                        (parameter.detach() - initial).square().sum()
                        for parameter, initial in zip(
                            g.policy.parameters(), initial_parameters
                        )
                    )
                )
            ),
        )

        if not all(math.isfinite(value) for value in obs.values()):
            raise FloatingPointError("Nonfinite exact evaluation")

        return obs


def evaluate_sampled(g, ref, beta, initial_parameters):
    """Estimate forward KL using policy samples and transfer-matrix logZ."""
    with preserve_rng(), torch.no_grad():
        torch.manual_seed(g.eval_seed + 100003 * g.eval_index)
        g.eval_index += 1

        logqs = []
        energies = []
        entropies = []
        magnetizations = []

        for start in range(0, g.eval_samples, g.eval_chunk):
            n = min(g.eval_chunk, g.eval_samples - start)
            acts = g._sample_actions_batch(n, eps=0.0)
            lp, entropy, spins = g._replay_batch(acts)

            logqs.append(lp.double())
            energies.append(g.env.energy_batch(spins).double())
            entropies.append(entropy.double())
            magnetizations.append(spins.double().mean(1))

        lq = torch.cat(logqs)
        E = torch.cat(energies)
        m = torch.cat(magnetizations)

        values = lq + E / g.T + ref["logZ"]
        n = len(values)
        mean_abs = m.abs().mean()
        m2 = m.square().mean()
        current_logZ = logZ_transfer(g.env.N, beta / g.T)

        obs = dict(
            kl_qp=float(values.mean()),
            kl_qp_mc_se=float(
                values.std(unbiased=True) / math.sqrt(n)
            ),
            evaluation_samples=n,
            training_target_kl_qp=float(
                (lq + beta * E / g.T + current_logZ).mean()
            ),
            energy_per_spin=float(E.mean() / g.env.D),
            energy_error_per_spin=float(
                E.mean() / g.env.D
                - ref["target_observables"]["energy_per_spin"]
            ),
            abs_m=float(mean_abs),
            chi=float(g.env.D * (m2 - mean_abs.square()) / g.T),
            frac_m_pos=float((m > 0).double().mean()),
            frac_m_zero=float((m == 0).double().mean()),
            U4=float(
                1 - m.pow(4).mean() / (3 * m2.square() + 1e-300)
            ),
            mean_action_entropy=float(torch.cat(entropies).mean()),
            mode_imbalance=abs(float((m > 0).double().mean()) - 0.5),
            parameter_displacement=float(
                torch.sqrt(
                    sum(
                        (parameter.detach() - initial).square().sum()
                        for parameter, initial in zip(
                            g.policy.parameters(), initial_parameters
                        )
                    )
                )
            ),
        )

        if not all(math.isfinite(value) for value in obs.values()):
            raise FloatingPointError("Nonfinite sampled evaluation")

        return obs


def evaluate_without_normalizer(g, initial_parameters):
    """Evaluate available metrics when no validated target logZ exists."""
    with preserve_rng(), torch.no_grad():
        torch.manual_seed(g.eval_seed + 100003 * g.eval_index)
        g.eval_index += 1

        energies = []
        magnetizations = []
        entropies = []

        for start in range(0, g.eval_samples, g.eval_chunk):
            n = min(g.eval_chunk, g.eval_samples - start)
            actions = g._sample_actions_batch(n, eps=0.0)
            _, entropy, spins = g._replay_batch(actions)

            energies.append(g.env.energy_batch(spins).double())
            magnetizations.append(spins.double().mean(1))
            entropies.append(entropy.double())

        E = torch.cat(energies)
        m = torch.cat(magnetizations)
        displacement = torch.sqrt(
            sum(
                (parameter.detach() - initial).square().sum()
                for parameter, initial in zip(
                    g.policy.parameters(), initial_parameters
                )
            )
        )

        return dict(
            kl_qp=None,
            kl_qp_mc_se=None,
            training_target_kl_qp=None,
            kl_unavailable_reason="No validated target logZ backend",
            evaluation_samples=g.eval_samples,
            energy_per_spin=float(E.mean() / g.env.D),
            abs_m=float(m.abs().mean()),
            mean_action_entropy=float(torch.cat(entropies).mean()),
            parameter_displacement=float(displacement),
        )


def make_evaluator(g, *, seed=0, eval_samples=2048, eval_chunk=256):
    if eval_samples < 2 or eval_chunk < 1:
        raise ValueError("eval_samples >= 2 and eval_chunk >= 1 required")

    initial = [
        parameter.detach().clone()
        for parameter in g.policy.parameters()
    ]

    g.eval_samples = eval_samples
    g.eval_chunk = eval_chunk
    g.eval_seed = seed + 271800
    g.eval_index = 0

    if g.env.D <= 16:
        ref = exact_reference(g.env, g.T, device=g.device)
        backend = "exact"
    elif g.env.kind == "ising" and g.env.N == 5:
        ref = ising_reference(g.env, g.T)
        backend = "sampled_transfer"
    else:
        ref = None
        backend = "sampled_no_normalizer"

    def evaluate(beta):
        if backend == "exact":
            obs = evaluate_exact(g, ref, beta, initial)
            obs.update(
                kl_qp_mc_se=0.0,
                evaluation_samples=len(ref["acts"]),
            )
        elif backend == "sampled_transfer":
            obs = evaluate_sampled(g, ref, beta, initial)
        else:
            obs = evaluate_without_normalizer(g, initial)

        return dict(evaluation_backend=backend, **obs)

    return evaluate, ref, initial