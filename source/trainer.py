import torch

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from torch import autograd

import time
import os
import fire
import random

import numpy as np

from source.simulators import NonlinearSSM
from source.proposal_models import NASMCProposal
from source.datasets import SSMDataset
from source.smc import smc_ssm
from source.models import SSM, POMDP

import multiprocessing


class NASMCTrainer:
    def run(self,
            run_dir: str = './runs/',
            proposal_lr: float = 1e-4,
            model_lr: float = 1e-4,
            num_steps: int = 1,
            save_decimation: int = 100,
            num_particles: int = 1000,
            sequence_length: int = 50,
            batch_size: int = 12,
            device_name: str = "cuda" if torch.cuda.is_available() else "cpu",
            seed: int = 95):
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        os.makedirs(run_dir, exist_ok=True)
        checkpoint_path = os.path.join(run_dir, 'checkpoint.pt')

        device = torch.device(device_name)

        proposal = NASMCProposal(4)
        model = SSM(4)
        simulator = NonlinearSSM()

        proposal_optimizer = torch.optim.Adam(proposal.parameters(), lr=proposal_lr)
        model_optimizer = torch.optim.Adam(model.parameters(), lr=model_lr)

        step = 1

        proposal.to(device)
        model.to(device)
        simulator.to(device)

        log_dir = None
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            proposal.load_state_dict(checkpoint['proposal'])
            model.load_state_dict(checkpoint['model'])
            simulator.load_state_dict(checkpoint['simulator'])
            proposal_optimizer.load_state_dict(checkpoint['proposal_optimizer'])
            step = checkpoint['step']
            num_particles = checkpoint['num_particles']
            sequence_length = checkpoint['sequence_length']
            log_dir = checkpoint['log_dir']

        summary_writer = SummaryWriter(log_dir)

        proposal.train()
        model.train()

        multiprocessing.set_start_method('spawn')
        dl = DataLoader(SSMDataset(simulator, sequence_length, batch_size), batch_size=None,
                        num_workers=2)
        start_time = time.time()
        for i, example in zip(range(num_steps), dl):
            observations = example.observations
            observations = observations.to(device)

            smc_result = smc_ssm(proposal, model, observations,
                                           num_particles)

            # Proposal Training

            # proposal_loss = -torch.sum(
            #     smc_result.intermediate_weights.detach() *
            #     smc_result.intermediate_proposal_log_probs) / batch_size
            proposal_loss = -torch.sum(
                smc_result.final_weights.detach() *
                smc_result.final_proposal_log_probs.squeeze(-1)) / batch_size

            proposal_optimizer.zero_grad()
            proposal_loss.backward()
            proposal_optimizer.step()

            # Model Training

            # model_loss = -torch.sum(
            #     smc_result.intermediate_weights.detach() *
            #     smc_result.intermediate_model_log_probs) / batch_size
            model_loss = -torch.sum(
                smc_result.final_weights.detach() *
                smc_result.final_model_log_probs.squeeze(-1)) / batch_size

            model_optimizer.zero_grad()
            model_loss.backward()
            model_optimizer.step()

            # Recording

            proposal_grad_norm = 0.0
            for param in proposal.parameters():
                if param.grad == None:
                    continue
                proposal_grad_norm = max(proposal_grad_norm, torch.max(torch.abs(param.grad.data)).item())

            model_grad_norm = 0.0
            for param in model.parameters():
                if param.grad == None:
                    continue
                model_grad_norm = max(model_grad_norm, torch.max(torch.abs(param.grad.data)).item())

            summary_writer.add_scalar('proposal_loss/train', proposal_loss, step)
            summary_writer.add_scalar('proposal_gradient', proposal_grad_norm, step)
            summary_writer.add_scalar('model_loss/train', model_loss, step)
            summary_writer.add_scalar('model_gradient', model_grad_norm, step)

            print(f'time = {time.time()-start_time:.1f} step = {step}  proposal_loss = {proposal_loss.item():.1f}  proposal_gradient = {proposal_grad_norm:.1f}  model_loss = {model_loss.item():.1f}  model_gradient = {model_grad_norm:.1f}')

            step += 1
            if step % save_decimation == 0:
                torch.save(
                    dict(proposal=proposal.state_dict(),
                         model=model.state_dict(),
                         simulator=simulator.state_dict(),
                         proposal_optimizer=proposal_optimizer.state_dict(),
                         step=step,
                         num_particles=num_particles,
                         sequence_length=sequence_length,
                         log_dir=summary_writer.log_dir), checkpoint_path)

        summary_writer.flush()

        torch.save(
            dict(proposal=proposal.state_dict(),
                 model=model.state_dict(),
                 simulator=simulator.state_dict(),
                 proposal_optimizer=proposal_optimizer.state_dict(),
                 step=step,
                 num_particles=num_particles,
                 sequence_length=sequence_length,
                 log_dir=summary_writer.log_dir), checkpoint_path)


if __name__ == '__main__':
    fire.Fire(NASMCTrainer)
