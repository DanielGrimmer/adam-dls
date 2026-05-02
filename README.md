# Adam-DLS: A Darwinian Lineage Simulation Optimizer

**Adam-DLS** is a PyTorch optimizer derived from evolutionary first principles. It modifies the Adam optimizer in three principled ways (detailed below) each derived from the asexual Fisher–Wright equivalence proved in the accompanying paper. After this minor surgery, Adam becomes a scientifically valid simulations of Darwinian evolution *in silico*. 

> **Paper:** [Direct From Darwin: Deriving Advanced Optimizers From Evolutionary First Principles](https://arxiv.org/abs/XXXX.XXXXX)  
> Daniel Grimmer · Philosophy, Yale University · *Evolutionary Computation* (submitted May, 2026)

---

## Abstract:

Evolutionary computation has long promised to deliver both high-performance optimization tools as well as rigorous scientific simulations of Darwinian evolution. However, modern algorithms frequently abandon evolutionary fidelity for physics-inspired heuristics or superficial biological metaphors. This paper derives advanced optimization algorithms directly from the evolutionary first principles. We introduce Darwinian Lineage Simulations (DLS) to prove that, in an asexual context, Fisher's and Wright's historically opposed views of evolution are actually formally equivalent. This unification requires carefully partitioning Fisher's deterministically-evolving total population into Wright's randomly-drifting sub-populations. We prove that proper bookkeeping requires introducing a specific kind of structured noise (the DLS noise relation). Crucially, however, *any bookkeeping choices* which satisfy this relation will result in a faithful simulation of evolution. Using this vast representational freedom, we prove that a broad family of battle-tested optimization algorithms are already perfectly compatible with our evolutionary dynamics. These include: Stochastic Gradient Ascent, Natural Gradient Descent, and the Damped Newton's method among many others. Simply adding DLS noise (i.e., evolutionarily faithful genetic drift), these algorithms become scientifically valid *in silico* simulations of Darwinian evolution. Finally, we demonstrate that even the state-of-the-art Adam optimizer can be brought into evolutionary compliance through a minor, principled mathematical surgery.

## Overview:

Adam-DLS modifies vanilla Adam in three ways to make it consistent with Fisher's and Wright's theories of evolution. Namely:

1. **The DLS Model of Genetic Drift** — Under Fisherian dynamics, the population's current variance, `V_g`,  acts as a pre-conditioner on the log-fitness gradient. From Wright, we understand that genetic drift comes from sampling smaller populations or individuals out of a larger distribution. From these two considerations one can derive the *DLS noise relation*: `W_g = μ²I − (V_{g+1} − V_g)` which ties the covariance matrix of Gaussian noise, `W_g`, to the mutation rate, `μ²`, and changes in the algorithm's preconditioner `V_{g+1} − V_g`. This particular noise structure is not optional; genetic drift must look exactly like this if one's simulation is to be evolutionarily faithful.
2. **Global momentum alignment scalar** (*d*) — Adam's momentum term is rescaled by a scalar which measures the alignment between the current gradient and the accumulated momentum. This converts Adam's additive momentum (which is evolutionarily non-compliant) into a non-diagonal preconditioner, `V_g`, with a rank-1 extension in the direction of the accumulated past gradients. Increasing in the population's variance in this direction correspondingly increases the selection pressure in this direction.
3. **Non-trivial second-moment initialization** — the second moment is seeded at `s₀ = (1 − β₂) · f₀²` rather than zero, to be consistent with our method of evolutionary bookkeeping.

---

## Installation

```bash
git clone https://github.com/danielgrimmer/adam-dls.git
cd adam-dls
pip install -r requirements.txt
```

A PyPI release is planned following journal publication.

---

## Quick Start

```python
import torch
from adam_dls import AdamDLS

model = MyModel()
optimizer = AdamDLS(
    model.parameters(),
    lr=2e-3,
    betas=(0.9, 0.999),
    eps=1e-8,
    mu_sq=2e-4,   # mutation rate squared (controls drift magnitude)
    delta=0,      # soft-error floor (set > 0 to enforce minimum variance)
    DLS_noise=True
)

for inputs, targets in dataloader:
    optimizer.zero_grad()
    loss = criterion(model(inputs), targets)
    loss.backward()
    optimizer.step()
```

To run Adam-DLS **without** the stochastic drift (i.e., only the momentum correction), set `DLS_noise=False`. This isolates the effect of the global alignment scalar *d* from the genetic drift noise.

---

## Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lr` | `2e-3` | Learning rate |
| `betas` | `(0.9, 0.999)` | Momentum decay rates (same as Adam) |
| `eps` | `1e-8` | Numerical stability term (same as Adam) |
| `mu_sq` | `2e-4` | Mutation rate squared; controls the base magnitude of DLS noise |
| `delta` | `0` | Soft-error floor; triggers a variance spike if minimum noise variance falls below this value |
| `DLS_noise` | `True` | Enable/disable the stochastic drift term |

---

## Benchmarks

### Rosenbrock (2D)

The Rosenbrock function `f(x,y) = (a−x)² + b(y−x²)²` with `a=2, b=100` is a standard non-convex benchmark. Adam-DLS converges reliably to the global minimum at `(2, 4)` from arbitrary initializations.

```bash
python benchmarks/rosenbrock.py
```

### CIFAR-100 (CNN)

A simple 3-layer CNN on CIFAR-100 demonstrates that Adam-DLS scales to higher-dimensional parameter spaces and more complex architectures. Adam-DLS runs at approximately half the wall-clock speed of Adam on this benchmark; this overhead is intrinsic to the global noise generation and is quantified in the paper.

```bash
python benchmarks/cifar100_cnn.py
```

---

## Theoretical Background

The optimizer is grounded in the **asexual Fisher–Wright equivalence**: the theorem that Fisher's deterministic mass selection on the total population is formally equivalent to Wright's randomly drifting sub-populations, provided the DLS noise relation is satisfied. This equivalence is proved in the paper and implies that any optimizer satisfying the DLS noise relation constitutes a scientifically valid *in silico* simulation of Darwinian evolution.

For the full derivation — including the Price equation, Gaussian lineage propagation, and the down-sampling procedure that introduces genetic drift — see the paper linked above.

---

## Citation

If you use Adam-DLS in your research, please cite:

```bibtex
@article{grimmer2026darwin,
  title     = {Direct From Darwin: Deriving Advanced Optimizers From Evolutionary First Principles},
  author    = {Grimmer, Daniel},
  journal   = {Evolutionary Computation},
  year      = {2026},
  note      = {arXiv:XXXX.XXXXX [cs.NE]},
  url       = {https://arxiv.org/abs/XXXX.XXXXX}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
