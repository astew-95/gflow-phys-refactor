def make_batch_size_schedule(batch_start, batch_end):
    """
    Linearly ramp the training batch size from batch_start to batch_end
    over the course of training. Rationale: early in training the policy
    is far from converged, so smaller, cheaper, noisier batches cover
    more distinct trajectories per unit of compute; later, as the policy
    approaches the target, a larger batch gives a lower-variance gradient
    estimate, which matters more once you're trying to nail down a
    precise final distribution than it does early on (the same tradeoff
    behind "increase batch size instead of decaying the learning rate").
    This is specifically affordable now (and wasn't really before) BECAUSE
    of the batching fix above: since a whole batch is processed in one
    forward pass per timestep, increasing batch size doesn't multiply the
    number of forward CALLS the way it would have in the old per-
    trajectory loop -- it just makes each call's tensor a bit wider,
    which is close to free until you're memory-bound.
    """
    def fn(it, n_iter):
        frac = min(it / n_iter, 1.0)
        return int(round(batch_start + frac * (batch_end - batch_start)))
    return fn
