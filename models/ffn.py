import torch.nn as nn

class StepEmbedPolicy(nn.Module):
    """
    Batched, weight-shared-across-timesteps policy.


    Input per state: (masked spin vector [D], assignment mask [D],
    learned step embedding, and -- for grid models only -- normalized
    (row, col) of the site about to be chosen, a cheap partial nod
    toward giving the policy some sense of 2D locality without a full
    conv architecture). Output: 2 logits (for -1 / +1), regardless of D.
    """
    def __init__(self, D, n_actions=2, hidden=128, step_embed_dim=16, grid_N=None):
        super().__init__()
        self.D = D
        self.grid_N = grid_N
        extra = 2 if grid_N is not None else 0
        in_dim = D + D + step_embed_dim + extra
        self.step_embed = nn.Embedding(D, step_embed_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, n_actions),
        )
        if grid_N is not None:
            coords = torch.tensor(
                [[r / grid_N, c / grid_N] for r in range(grid_N) for c in range(grid_N)],
                dtype=torch.float32)
            self.register_buffer("coords", coords)  # [D, 2]

    def forward(self, spins, mask, step_t):
        """spins, mask: [B, D] tensors (same device/dtype). step_t: python
        int -- same timestep for the whole batch, since sampling advances
        one position at a time for every trajectory in lockstep (that's
        what makes the batching possible in the first place)."""
        B = spins.shape[0]
        step_idx = torch.full((B,), step_t, dtype=torch.long, device=spins.device)
        feats = [spins, mask, self.step_embed(step_idx)]
        if self.grid_N is not None:
            feats.append(self.coords[step_t].unsqueeze(0).expand(B, -1))
        return self.net(torch.cat(feats, dim=-1))  # [B, n_actions]

