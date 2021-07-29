import torch
import torch.nn as nn
import torch.distributions as D

from math import sqrt

from typing import NamedTuple


class SSMExample(NamedTuple):
    states: torch.Tensor
    observations: torch.Tensor


class SSMDataset(torch.utils.data.IterableDataset):
    def __init__(self, model: nn.Module, sequence_length: int = 1000, batch_size: int = 16):
        super().__init__()
        assert sequence_length > 0, 'sequence length must be greater than 0'
        self.sequence_length = sequence_length
        assert batch_size > 0, 'batch size must be greater than 0'
        self.batch_size = batch_size
        self.model = model

    def __iter__(self):
        while True:
            prior_distribution = self.model.prior()
            states = prior_distribution.sample((self.batch_size,))

            observation_sequence = []
            state_sequence = []
            for timestep in range(2, self.sequence_length + 2):
                observations = self.model.observation(states).sample()
                observation_sequence.append(observations)

                state_sequence.append(states)
                states = self.model.transition(states).sample()

            yield SSMExample(
                states=torch.stack(state_sequence, dim=1),
                observations=torch.stack(observation_sequence, dim=1))
