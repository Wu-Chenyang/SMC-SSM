
import torch
import torch.nn as nn
import torch.distributions as D

from typing import Tuple

from copy import deepcopy

class NASMCProposal(nn.Module):
    def __init__(self, state_dim: int = 2, obs_dim: int = 1, mixture_num: int = 3, hidden_dim: int = 50, lstm_num: int = 1):
        super().__init__()
        self.mixture_num = mixture_num
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        self.lstm_num = lstm_num
        
        self.lstm = nn.LSTM(obs_dim + state_dim, hidden_dim, lstm_num)
        self.h0 = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim * lstm_num),
            nn.Tanh(),
            nn.Linear(hidden_dim * lstm_num, hidden_dim * lstm_num),
            nn.Tanh()
        )
        self.c0 = deepcopy(self.h0)

        self.mixture = nn.Sequential(nn.Linear(hidden_dim, mixture_num), nn.Softmax(dim=-1))
        self.mean = nn.Linear(hidden_dim, mixture_num * state_dim)
        self.std = nn.Sequential(nn.Linear(hidden_dim, mixture_num * state_dim), nn.Softplus())
    
    def get_distribution(self, features: torch.Tensor) -> D.Distribution:
        mixtures = self.mixture(features)
        means = self.mean(features).reshape((features.shape[:-1] + (self.mixture_num, self.state_dim)))
        stds = self.std(features).reshape((features.shape[:-1] + (self.mixture_num, self.state_dim)))

        return D.MixtureSameFamily(
            D.Categorical(probs=mixtures),
            D.Independent(D.Normal(means, stds), 1)
        )
    
    def prior_proposal(self, observation: torch.Tensor) -> Tuple[D.Distribution, Tuple[torch.Tensor, torch.Tensor]]:
        h = self.h0(observation).reshape(observation.shape[:-1] + (self.hidden_dim, self.lstm_num)).moveaxis(-1, 0)
        c = self.c0(observation).reshape(observation.shape[:-1] + (self.hidden_dim, self.lstm_num)).moveaxis(-1, 0)
        return self.get_distribution(h[-1]), (h,c)

    def transition_proposal(self, previous_state: torch.Tensor, observation: torch.Tensor, lstm_state: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[D.Distribution, Tuple[torch.Tensor, torch.Tensor]]:
        batch_shape = observation.shape[:-1]
        inputs = torch.cat((previous_state, observation), axis=-1)
        inputs = inputs.reshape(1, -1, inputs.shape[-1])
        h = lstm_state[0].reshape(lstm_state[0].shape[0], -1, lstm_state[0].shape[-1])
        c = lstm_state[1].reshape(lstm_state[1].shape[0], -1, lstm_state[1].shape[-1])
        output, lstm_state = self.lstm(inputs, (h,c))
        h = lstm_state[0].reshape((lstm_state[0].shape[0],) + batch_shape + (lstm_state[0].shape[-1],))
        c = lstm_state[1].reshape((lstm_state[1].shape[0],) + batch_shape + (lstm_state[1].shape[-1],))
        return self.get_distribution(output.reshape(batch_shape + (output.shape[-1],))), (h, c)