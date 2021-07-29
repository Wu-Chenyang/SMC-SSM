import torch
import torch.nn as nn
import torch.distributions as D

from math import sqrt

class NonlinearSSM(nn.Module):
    # An example for nonlinear state space model.
    def __init__(self, trans_scale: float = sqrt(10.), obs_scale: float = 1.0, device: str = None):
        super().__init__()
        self.prior_mean = nn.Parameter(torch.zeros(2, device=device), requires_grad=False)
        self.prior_scale = nn.Parameter(torch.diag(torch.tensor([sqrt(5), 1e-4], device=device)), requires_grad=False)
        self.trans_scale = nn.Parameter(torch.diag(torch.tensor([trans_scale, 1e-4], device=device)), requires_grad=False)
        self.obs_scale = obs_scale

    def prior(self) -> D.Distribution:
        return D.MultivariateNormal(self.prior_mean, scale_tril=self.prior_scale)
    
    def transition(self, state: torch.Tensor) -> D.Distribution:
        mean = torch.stack([state[..., 0] / 2. + 25 * state[..., 0] / (1 + state[..., 0] * state[..., 0]) + 8 * torch.cos(1.2 * state[..., 1]), state[..., 1] + 1], dim=-1)
        return D.MultivariateNormal(mean, scale_tril=self.trans_scale)

    def observation(self, state: torch.Tensor) -> D.Distribution:
        state = state[..., 0, None]
        return D.Independent(D.Normal(state * state / 20., self.obs_scale), 1)