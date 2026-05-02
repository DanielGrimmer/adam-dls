# baselines/noisy_adam.py
#NoisyAdam is a baseline comparison optimizer used in benchmarking. It is not part of the Adam-DLS package and is not intended for production use.

import math
import torch
from torch.optim import Optimizer

class NoisyAdam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 mu_sq=1e-4, just_SGD=False, minimize=True):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate (lr): {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= mu_sq:
            raise ValueError(f"Invalid mutation rate: {mu_sq}")            

        defaults = dict(lr=lr, betas=betas, eps=eps, mu_sq=mu_sq, just_SGD=just_SGD, minimize=minimize)
        super(NoisyAdam, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            mu_sq = group['mu_sq']
            
            for p in group['params']:
                if p.grad is None:
                    continue

                f_g = p.grad
                state = self.state[p]

                # State initialization (with s_0 = 0)
                if len(state) == 0:
                    # Note: PyTorch step 1 corresponds to using generation g=0 to calculate g=1
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['s'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                state['step'] += 1
                step = state['step']

                # Retrieve g-th generation states
                m_g = state['m']
                s_g = state['s']

                # --- 1. Compute g+1 moments ---
                m_g_plus_1 = beta1 * m_g + (1 - beta1) * f_g
                s_g_plus_1 = beta2 * s_g + (1 - beta2) * f_g ** 2
                hats_g_plus_1 = s_g_plus_1 / (1 - beta2 ** step)

                # --- 3. Calculate the pre-conditioner D_g and apply deterministic update ---.
                D_g = (lr / (hats_g_plus_1.sqrt() + eps)) / (1 - beta1 ** step)
                
                if group['just_SGD']:
                    update_direction = lr * f_g
                else:
                    update_direction = D_g * m_g_plus_1

                if group['minimize']:
                    p.sub_(update_direction)
                else:
                    p.add_(update_direction)
                
                # --- 4. Generate Naive Noise (\xi_g) ---
                xi_g = math.sqrt(mu_sq) * torch.randn_like(p)
                p.add_(xi_g)

                # --- 5. Update states for next generation ---
                state['m'].copy_(m_g_plus_1)
                state['s'].copy_(s_g_plus_1)

        return loss
