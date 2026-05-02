# Adam-DLS: A Darwinian Lineage Simulation Optimizer

**Adam-DLS** is a PyTorch optimizer derived from evolutionary first principles. It modifies the Adam optimizer in three principled ways (detailed below) each derived from the asexual Fisher–Wright equivalence proved in the accompanying paper. After this minor surgery, the Adam optimizer becomes a scientifically valid simulations of Darwinian evolution *in silico*. 

> **Paper:** Direct From Darwin: Deriving Advanced Optimizers From Evolutionary First Principles *(arXiv link coming soon)*  
> Daniel Grimmer · Philosophy, Yale University · *Evolutionary Computation* (submitted May, 2026)

---

## Overview:

Adam-DLS is a PyTorch optimizer derived from evolutionary biology rather than engineering heuristics. Most "evolutionary" algorithms in machine learning are built upon superficial biological metaphors or statistical physics heuristics; by contrast, this one is grounded in *actual population genetics*. The key theoretical result is a formal equivalence between Fisher's deterministic view of evolution (mass selection acting on a whole population) and Wright's stochastic view (small sub-populations drifting randomly across a fitness landscape) — unified here through a precise noise structure called *the DLS noise relation*. This relation turns out to be the missing ingredient that makes many standard optimizers — including Stochastic Gradient Ascent, Natural Gradient Descent, and Damped Newton's method — into scientifically valid *in silico* simulations of Darwinian evolution. The Adam optimizer requires one additional fix: its momentum term is replaced by a principled rank-1 correction that aligns historical gradient information with current selection pressure. The result, Adam-DLS, is a drop-in replacement for Adam that doubles as a genuine computational model of Darwinian adaptation.

Adam-DLS modifies vanilla Adam in three ways to make it consistent with Fisher's and Wright's theories of evolution. Namely:

1. **The DLS Model of Genetic Drift** — Under Fisherian dynamics, the population's current variance, `V_g`,  acts as a pre-conditioner on the log-fitness gradient. From Wright, we understand that genetic drift comes from sampling smaller populations or individuals out of a larger distribution. From these two considerations one can derive the *DLS noise relation*: `W_g = μ²I − (V_{g+1} − V_g)` which ties the covariance matrix of Gaussian noise, `W_g`, to the mutation rate, `μ²`, and changes in the algorithm's preconditioner `V_{g+1} − V_g`. This particular noise structure is not optional; genetic drift must look exactly like this if one's simulation is to be evolutionarily faithful.
2. **Global momentum alignment scalar** (*d*) — Adam's momentum term is rescaled by a scalar which measures the alignment between the current log-fitness gradient and the accumulated momentum. This converts Adam's additive momentum (which is evolutionarily non-compliant) into a non-diagonal preconditioner, `V_g`, with a rank-1 extension in the direction of the accumulated past gradients. Increasing in the population's variance in this direction correspondingly increases the selection pressure in this direction.
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
    lr=1e-3,
    betas=(0.9, 0.999),
    eps=1e-8,
    mu_sq=1e-4,   # mutation rate squared (controls magnitude of genetic drift)
    delta=0,      # soft-error floor (set > 0 to enforce minimum variance)
)

for inputs, targets in dataloader:
    optimizer.zero_grad()
    loss = criterion(model(inputs), targets)
    loss.backward()
    optimizer.step()
```

---

## Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lr` | `1e-3` | Learning rate (same as Adam) |
| `betas` | `(0.9, 0.999)` | Momentum decay rates (same as Adam) |
| `eps` | `1e-8` | Numerical stability term (same as Adam) |
| `mu_sq` | `1e-4` | Mutation rate squared; controls the base magnitude of DLS noise |
| `delta` | `0` | Soft-error floor; triggers an ad hoc mutation spike if the minimum noise variance falls below this value |

---

## Benchmarks

### Rosenbrock (2D)

The Rosenbrock function `f(x,y) = (a−x)² + b(y−x²)²` with `a=2, b=100` is a standard non-convex benchmark. Adam-DLS converges reliably to the global minimum at `(2, 4)` just as the unmodified Adam operator does.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/danielgrimmer/adam-dls/blob/main/benchmarks/Rosenbrock_benchmark.ipynb)

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
