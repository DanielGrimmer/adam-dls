# baselines/optimizer.py

import math
import torch
from torch.optim import Optimizer


class NoisyAdam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 mu_sq=2e-4, just_SGD=False, record_history=False):
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

        defaults = dict(lr=lr, betas=betas, eps=eps, mu_sq=mu_sq, just_SGD=just_SGD,record_history=record_history)
        super(NoisyAdam, self).__init__(params, defaults)

        # Trackers for plotting
        self.Dm_g_history = []

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

                # State initialization (incorporating s_0 = 0)
                if len(state) == 0:
                    # Note: PyTorch step 1 mathematically corresponds to calculating g=1 from g=0
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

                # --- 3. Calculate D_g ---
                # g = step - 1. Therefore, exponent on D_g is step, and on D_g+1 is step + 1.
                D_g = (lr / (hats_g_plus_1.sqrt() + eps)) / (1 - beta1 ** step)

                # Track D_g m_g for Adam analysis
                Dm_g = D_g * m_g
                if group['record_history']: self.Dm_g_history.append(Dm_g.clone()) # Added for tracking

                # --- 4. Generate Naive Noise (\xi_g) ---

                xi_g = math.sqrt(mu_sq) * torch.randn_like(p)

                # We use .sub_() to subtract the gradient step
                if group['just_SGD']:
                    p.sub_(lr * f_g)
                else:
                    p.sub_(D_g * m_g_plus_1)

                p.add_(xi_g)

                # --- 6. Update states for next generation ---
                state['m'].copy_(m_g_plus_1)
                state['s'].copy_(s_g_plus_1)

        return loss
