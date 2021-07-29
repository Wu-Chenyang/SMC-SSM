import torch
import torch.nn as nn
import torch.distributions as D

from typing import Tuple

class CGMM(nn.Module):
    # Conditional Gaussian Mixture Model
    # Yield an Gaussian mixture distribution conditioned on the input.
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, mixture_num: int):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        self.mixture = nn.Sequential(nn.Linear(hidden_dim, mixture_num), nn.Softmax(dim=-1))
        self.mean = nn.Linear(hidden_dim, mixture_num * output_dim)
        self.std = nn.Sequential(nn.Linear(hidden_dim, mixture_num * output_dim), nn.Softplus())

        self.mixture_num = mixture_num
        self.output_dim = output_dim
    
    def forward(self, inputs: torch.Tensor) -> D.Distribution:
        features = self.feature(inputs)
        mixtures = self.mixture(features)
        means = self.mean(features).reshape((features.shape[:-1] + (self.mixture_num, self.output_dim)))
        stds = self.std(features).reshape((features.shape[:-1] + (self.mixture_num, self.output_dim)))

        return D.MixtureSameFamily(
            D.Categorical(probs=mixtures),
            D.Independent(D.Normal(means, stds), 1)
        )

class SSM(nn.Module):
    def __init__(self, state_dim: int = 2, obs_dim: int = 1,
                trans_mixture_num: int = 2, trans_hidden_dim: int = 50,
                obs_mixture_num: int = 2, obs_hidden_dim: int = 50, device=None):
        super().__init__()
        self.prior_mean = nn.Parameter(torch.zeros(state_dim, device=device), requires_grad=False)
        self.prior_scale = nn.Parameter(torch.eye(state_dim, device=device), requires_grad=False)
        self.trans_net = CGMM(state_dim, state_dim, trans_hidden_dim, trans_mixture_num)
        self.obs_net = CGMM(state_dim, obs_dim, obs_hidden_dim, obs_mixture_num)

    def prior(self) -> D.Distribution:
        return D.MultivariateNormal(self.prior_mean, scale_tril=self.prior_scale)
    
    def transition(self, state: torch.Tensor) -> D.Distribution:
        return self.trans_net(state)

    def observation(self, state: torch.Tensor) -> D.Distribution:
        return self.obs_net(state)