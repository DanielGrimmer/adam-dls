# adam_dls/optimizer.py

import math
import torch
from torch.optim import Optimizer


class AdamDLS(Optimizer):
    """
    Direct From Darwin: Adam-DLS (Darwinian Lineage Simulation)

    An evolutionarily faithful version of the Adam optimizer derived from evolutionary first principles.
    Modifications from vanilla Adam:
    1. Index shift on second moments & non-trivial initialization (s_0 = (1 - beta2) * f_0^2).
    2. Rescaling of momentum term based on alignment of current and past gradient.
    3. Injection of scientifically accurate genetic drift (DLS noise) applied globally.
    4. Variance limits / Soft-Error handling to respect biological speed limits.
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 mu_sq=1e-4, delta=0, record_history=False, minimize=False):
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
        if not 0.0 <= delta:
            raise ValueError(f"Invalid delta value: {delta}")

        defaults = dict(lr=lr, betas=betas, eps=eps, mu_sq=mu_sq,
                        delta=delta, record_history=record_history, minimize=minimize)
        super(AdamDLS, self).__init__(params, defaults)

        if record_history:
            self.D_g_history = []
            self.m_g_history = []
            self.d_history = []
            self.mu_history = []

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Phase 1 Setup: Initialize lists to gather global data state
        p_list = []
        f_list = []
        m_list = []
        m_p1_list = []
        D_list = []
        D_p1_list = []
        state_list = []
        s_p1_list = []
        beta1_list = []

        # The environment (mu_sq, delta) and core time-constant (beta1) apply globally for noise
        mu_sq_global = self.defaults['mu_sq']
        delta_global = self.defaults['delta']
        beta1_global = self.defaults['betas'][0]

        # --- Phase 1: Local Gather ---
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue

                f_g = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['s'] = (1 - beta2) * f_g.clone() ** 2

                state['step'] += 1
                step = state['step']

                m_g = state['m']
                s_g = state['s']

                # 1. Calculate Vanilla Adam's Preconditioner D_g
                s_g_debiased = s_g / (1 - beta2 ** step)
                D_g = (lr / (s_g_debiased.sqrt() + eps)) / (1 - beta1 ** step)

                # 2. Compute Next Step's Moments and Preconditioner D_g_plus_1
                m_g_plus_1 = beta1 * m_g + (1 - beta1) * f_g
                s_g_plus_1 = beta2 * s_g + (1 - beta2) * f_g ** 2
                s_g_plus_1_debiased = s_g_plus_1 / (1 - beta2 ** (step + 1))
                D_g_plus_1 = (lr / (s_g_plus_1_debiased.sqrt() + eps)) / (1 - beta1 ** (step + 1))

                # 3. Store components for global stochastic noise generation AND global d_g
                p_list.append(p)
                f_list.append(f_g)
                m_list.append(m_g)
                m_p1_list.append(m_g_plus_1)
                D_list.append(D_g)
                D_p1_list.append(D_g_plus_1)
                state_list.append(state)
                s_p1_list.append(s_g_plus_1)
                beta1_list.append(beta1)

        # Skip global assessment if no gradients were found
        if len(p_list) == 0:
            return loss

        # --- Phase 2: Global Assessment, Holistic Momentum Scalar & Noise Generation ---
        # Flatten the entire genotype into global tensors
        f_flat = torch.cat([f.view(-1) for f in f_list])
        m_flat = torch.cat([m.view(-1) for m in m_list])
        m_p1_flat = torch.cat([m.view(-1) for m in m_p1_list])
        D_flat = torch.cat([d.view(-1) for d in D_list])
        D_p1_flat = torch.cat([d.view(-1) for d in D_p1_list])

        if self.defaults['record_history']:
            self.D_g_history.append(D_flat.clone())
            self.m_g_history.append(m_flat.clone())

        # Calculate Global Momentum Scalar d_global
        D_m_flat = D_flat * m_flat
        m_T_D_m_global = torch.sum(m_flat * D_m_flat)
        num_global = torch.sum(f_flat * D_m_flat)

        d_global = torch.where(
            m_T_D_m_global > 1e-12,
            num_global / (m_T_D_m_global + 1e-15),
            torch.tensor(1.0, device=m_flat.device)
        )

        if self.defaults['record_history']:
            self.d_history.append(d_global.clone())

        # Generate global DLS genetic drift
        xi_global_flat = self._generate_dls_noise(
            m_flat, m_p1_flat, D_flat, D_p1_flat,
            beta1_global, delta_global, mu_sq_global
        )

        # --- Phase 3: Global Distribution ---
        # Slice the unified noise and apply deterministic + stochastic updates locally
        offset = 0
        for i, p in enumerate(p_list):
            numel = p.numel()
            beta1_local = beta1_list[i]
            f_g = f_list[i]
            m_g = m_list[i]
            D_g = D_list[i]

            # 1. Apply Deterministic Update (using global alignment d_global)
            update_direction = D_g * ((1 - beta1_local) * f_g + beta1_local * d_global * m_g)
            if self.defaults['minimize']:
                p.sub_(update_direction)
            else:
                p.add_(update_direction)

            # 2. Apply Stochastic Drift
            xi_p = xi_global_flat[offset : offset + numel].view_as(p)
            p.add_(xi_p)

            # 3. Commit the state updates
            state = state_list[i]
            state['m'].copy_(m_p1_list[i])
            state['s'].copy_(s_p1_list[i])

            offset += numel

        return loss

    def _generate_dls_noise(self, m_g_flat, m_g_p1_flat, D_g_flat, D_g_p1_flat, beta1, delta, mu_sq):
        
        # Fully vectorized mathematical components using +1e-15 to protect against GPU NaN halts
        mDm_g_scalar = torch.sum(m_g_flat * D_g_flat * m_g_flat)
        safe_denominator = torch.sqrt(mDm_g_scalar + 1e-15)
        y_g_calculated = math.sqrt(beta1) * (D_g_flat * m_g_flat) / safe_denominator
        y_g = torch.where(
            mDm_g_scalar > 1e-12,
            y_g_calculated,
            torch.zeros_like(m_g_flat)
        )

        mDm_g_p1_scalar = torch.sum(m_g_p1_flat * D_g_p1_flat * m_g_p1_flat)
        safe_denominator_p1 = torch.sqrt(mDm_g_p1_scalar + 1e-15)
        y_g_p1_calculated = math.sqrt(beta1) * (D_g_p1_flat * m_g_p1_flat) / safe_denominator_p1
        y_g_p1 = torch.where(
            mDm_g_p1_scalar > 1e-12,
            y_g_p1_calculated,
            torch.zeros_like(m_g_p1_flat)
        )

        # Compute minimum global variance constraint
        S_g = mu_sq - (1 - beta1) * (D_g_p1_flat - D_g_flat)

        s_min = torch.min(S_g)
        norm_y_g_p1_sq = torch.sum(y_g_p1 ** 2)

        # Calculate and apply potential Soft-Error spike
        deficit = delta - (s_min - norm_y_g_p1_sq)
        spike = torch.clamp(deficit, min=0.0)
        mu_sq_spike = mu_sq + spike
        S_g = mu_sq_spike - (1 - beta1) * (D_g_p1_flat - D_g_flat)

        #This records the smallest mutation rate which would be necesary to cover this down-sampling
        if self.defaults['record_history']:
            self.mu_history.append(mu_sq + deficit)

        # Massive N x 2 decomposition
        S_g_sqrt_inv = 1.0 / torch.sqrt(S_g)
        U = torch.stack([S_g_sqrt_inv * y_g, S_g_sqrt_inv * y_g_p1], dim=1)

        # Note: For massive parameter tensors (N > 10M), this N x 2 QR decomposition
        # could be replaced with an explicit Gram-Schmidt step for optimization,
        # but torch.linalg.qr is maintained here for numerical stability and readability.
        Q, R = torch.linalg.qr(U, mode='reduced')

        C = torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=U.dtype, device=U.device)
        M = R @ C @ R.T
        L, E = torch.linalg.eigh(M)
        A = Q @ E
        K = torch.diag(torch.sqrt(torch.clamp(1.0 + L, min=0.0)) - 1.0)
        z = torch.randn_like(S_g)

        # O(N) Rank-2 Update: Highly efficient for massive networks!
        xi_g_flat = torch.sqrt(S_g) * (z + A @ (K @ (A.T @ z)))

        return xi_g_flat
