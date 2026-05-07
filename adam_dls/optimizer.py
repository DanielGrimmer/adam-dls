# adam_dls/optimizer.py

import math
import torch
from torch.optim import Optimizer
import warnings

class AdamDLS(Optimizer):
    """
    Direct From Darwin: Adam-DLS (Darwinian Lineage Simulation)

    An evolutionarily faithful version of the Adam optimizer derived from evolutionary first principles.
    Modifications from vanilla Adam:
    1. Injection of scientifically accurate genetic drift (DLS noise) applied globally. 
       Variance limits / Soft-Error handling to respect biological speed limits.    
    2. Rescaling of momentum term based on alignment of current and past gradient.
    3. Index shift on second moments & non-trivial initialization (s_0 = (1 - beta2) * f_0^2).

    To run Adam-DLS without noise, set naive_noise=true and mu_sq=0.
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 mu_sq=1e-4, delta=0, record_scalar_history=False, record_vector_history=False, naive_noise=False, minimize=True):
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

        defaults = dict(lr=lr, betas=betas, eps=eps, mu_sq=mu_sq, delta=delta,
                        record_scalar_history=record_scalar_history, record_vector_history=record_vector_history,
                        naive_noise=naive_noise, minimize=minimize)
        super(AdamDLS, self).__init__(params, defaults)

        if record_vector_history:
            self.D_g_history = []
            self.m_g_history = []
        if record_scalar_history:
            self.d_history = []
            self.mu_history = []

        self._noise_call_count = 0
        self._spike_count = 0
        self._max_spike = 0.0
    
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

        if self.defaults['record_vector_history']:
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

        if self.defaults['record_scalar_history']:
            self.d_history.append(d_global.clone())

        if self.defaults['naive_noise']:
            # Generate naive noise
            xi_global_flat = math.sqrt(mu_sq_global) * torch.randn_like(m_flat)
        else:
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

        """
        Sample genetic drift  ξ_g ~ N(0, W_g) for one Adam-DLS generation.
    
        Implements the DLS noise relation (Appendix A):
            W_g = μ²I − (V_{g+1} − V_g)
        where V_g is the Adam-DLS lineage variance.  Because W_g is built from a diagonal
        matrix S_g perturbed by a signed rank-2 update (+y_g y_g^T − y_{g+1} y_{g+1}^T),
        its matrix square root W_g^{1/2} is computed in O(N) time: a thin QR
        decomposition on the N×2 scaled-momentum matrix U reduces the problem to a
        trivial 2×2 eigendecomposition, after which the noise is applied as a
        rank-2 update to an isotropic base sample z ~ N(0, I).
    
        If W_g is not positive semi-definite — a 'soft error' caused by μ² being
        too small to accommodate the variance change V_{g+1} − V_g — a one-off
        mutation spike γ² is added to μ² to restore PSD-ness.  When record_scalar_history
        is enabled, the minimum μ² that would have avoided the spike at each step
        is recorded in self.mu_history; call check_soft_errors() after training to
        inspect these without incurring a per-step CPU-GPU sync.
    
        Args:
            m_g_flat    (N,): Flattened first-moment vector at generation g.
            m_g_p1_flat (N,): Flattened first-moment vector at generation g+1.
            D_g_flat    (N,): Flattened diagonal preconditioner at generation g.
            D_g_p1_flat (N,): Flattened diagonal preconditioner at generation g+1.
            beta1       (float): First-moment decay rate β₁.
            delta       (float): Soft-error safety floor δ ≥ 0.
            mu_sq       (float): Baseline mutation rate μ² ≥ 0.
    
        Returns:
            xi_g_flat   (N,): Sampled genetic drift ξ_g ~ N(0, W_g).
        """
        
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
        
        # This records the smallest mutation rate which would be necessary to cover for the down-sampling at this step
        # The resulting mu_history is therefore useful for setting a relatively small mutation rate that avoids soft errors (ad hoc mutation spikes)
        if self.defaults['record_scalar_history']:
            self.mu_history.append(mu_sq + deficit)

        # Track spike statistics and report every 1000 calls
        self._noise_call_count += 1
        spike_val = mu_sq_spike.item()
        if spike_val > mu_sq:
            self._spike_count += 1
            if spike_val > self._max_spike and self._spike_count > 5:
                self._max_spike = spike_val
                print(
                f"Soft-error spikes: {self._spike_count} / {self._noise_call_count} steps | "
                f"Largest spike: {self._max_spike:.3e}"
                )
        
        # N x 2 decomposition
        S_g_sqrt_inv = 1.0 / torch.sqrt(S_g)
        U = torch.stack([S_g_sqrt_inv * y_g, S_g_sqrt_inv * y_g_p1], dim=1)

        # Note: For high-dimensional parameter tensors (N > 10M), this N x 2 QR decomposition
        # could be replaced with an explicit Gram-Schmidt step for optimization,
        # but torch.linalg.qr is maintained here for numerical stability and readability.
        Q, R = torch.linalg.qr(U, mode='reduced')

        C = torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=U.dtype, device=U.device)
        M = R @ C @ R.T
        M = (M + M.T) / 2            #Guarantee Symmetry
        L, E = torch.linalg.eigh(M)
        A = Q @ E
        K = torch.diag(torch.sqrt(torch.clamp(1.0 + L, min=0.0)) - 1.0)
        z = torch.randn_like(S_g)

        # O(N) Rank-2 Update: Highly efficient for massive networks!
        xi_g_flat = torch.sqrt(S_g) * (z + A @ (K @ (A.T @ z)))

        return xi_g_flat

    def check_soft_errors(self):
        """
        Call after training to check whether any soft errors (mutation spikes)
        occurred. Triggers one CPU-GPU sync. Returns the number of steps
        where a spike was required and the maximum spike magnitude.
        """
        if not self.defaults['record_scalar_history']:
            raise RuntimeError("Enable record_scalar_history=True to check soft errors.")
        spikes = torch.stack(self.mu_history) - self.defaults['mu_sq']
        n_errors = int((spikes > 1e-12).sum().item())
        max_spike = spikes.max().item()
        if n_errors > 0:
            import warnings
            warnings.warn(
                f"Soft errors occurred at {n_errors} steps (max spike: {max_spike:.2e}). "
                f"Consider raising mu_sq above {self.defaults['mu_sq'] + max_spike:.2e}.",
                UserWarning
            )
        return n_errors, max_spike
