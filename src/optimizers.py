"""Optimizer candidates under comparison: AdamW, SGD+momentum, Nero, Muon.

AdamW and SGD are used directly from torch.optim — no reimplementation
needed or wanted. Nero and Muon are implemented from scratch below.

Muon is a from-scratch reproduction, documented below as simplified rather
than a verbatim port. Nero matches the authors' reference implementation
(https://github.com/jxbz/nero) closely -- an earlier version of this file
had a meaningfully different, buggy "simplified" Nero (no mean-centering,
wrong re-projection target, sum-vs-mean second moment, and spurious
momentum on 1D params that the real method never has); see git history for
what changed and why.

- Nero (Liu, Bernstein, Meister, Yue. "Learning by Turning: Neural
  Architecture-Aware Optimisation." ICML 2021): momentum-free. Each
  parameter with more than 1 dimension is immediately centered (mean
  subtracted) and projected to unit norm per neuron (output row) at
  construction time -- "projected gradient descent over the space of
  balanced networks," per the paper -- and re-centered/re-projected after
  every step. A per-neuron second-moment (RMS) running average normalizes
  the gradient; the step size is additionally scaled by a per-tensor
  constant fixed at construction (the mean pre-projection neuron norm).
  1D parameters (biases, norm affine params) use the same per-"neuron"
  machinery with neuron size 1 (norm = abs(value)) but are never centered
  -- centering a size-1 vector would just zero it, and the reference
  implementation explicitly disallows this.

- Muon (Jordan et al., "Muon: An optimizer for hidden layers in neural
  networks," 2024 — technical report, not peer-reviewed; not in any
  PyTorch version when this was written, added as torch.optim.Muon only
  in PyTorch 2.9): momentum, then approximate orthogonalization of the
  momentum matrix via Newton-Schulz iteration, applied only to 2D
  hidden-layer weight matrices. Following the reference design, this is a
  hybrid optimizer: parameters flagged `use_muon=False` in their param
  group (embeddings, output head, biases, norm params) instead receive a
  plain AdamW-style update in the same optimizer step. Checked against the
  now-official implementation: Newton-Schulz coefficients and the
  LR-adjustment formula match exactly, but this uses PLAIN (heavy-ball)
  momentum for the orthogonalization input, not native Muon's DEFAULT
  Nesterov momentum (`g_t + momentum * B_t` instead of just `B_t`) --
  matches its `nesterov=False` option, not its default. No weight decay
  is applied here either (native defaults to 0.1, decoupled). Neither
  difference has been tested for impact on this project's results.
"""
from typing import Iterable

import torch
from torch.optim import Optimizer


def zeropower_via_newtonschulz(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Approximate the orthogonalization of a 2D matrix (replace its
    singular values with 1, keep its singular vectors) via Newton-Schulz
    iteration, avoiding an explicit SVD. Coefficients are the quintic
    Newton-Schulz coefficients from the Muon reference implementation.
    """
    assert G.dim() == 2
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G / (G.norm() + eps)
    transpose = X.size(0) > X.size(1)
    if transpose:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transpose:
        X = X.T
    return X


def _neuron_norm(x: torch.Tensor) -> torch.Tensor:
    """Per-neuron (per-output-row) L2 norm, shaped to broadcast against x.
    For 1D tensors, each scalar is its own size-1 "neuron" (norm = abs
    value) -- matches the reference implementation exactly, and is what
    lets 1D params (biases, norm affine) go through the same machinery
    with no special-casing.
    """
    if x.dim() > 1:
        view_shape = [x.shape[0]] + [1] * (x.dim() - 1)
        return x.reshape(x.shape[0], -1).norm(dim=1).view(*view_shape)
    return x.abs()


def _neuron_mean(x: torch.Tensor) -> torch.Tensor:
    if x.dim() > 1:
        view_shape = [x.shape[0]] + [1] * (x.dim() - 1)
        return x.reshape(x.shape[0], -1).mean(dim=1).view(*view_shape)
    raise ValueError("neuron_mean is not defined for 1D tensors (would zero them out)")


class Nero(Optimizer):
    def __init__(self, params: Iterable[torch.nn.Parameter], lr: float = 0.01,
                 beta: float = 0.999, constraints: bool = True):
        defaults = dict(lr=lr, beta=beta, constraints=constraints)
        super().__init__(params, defaults)

        with torch.no_grad():
            for group in self.param_groups:
                for p in group["params"]:
                    if group["constraints"] and p.dim() > 1:
                        p.data -= _neuron_mean(p)
                        p.data /= _neuron_norm(p)
                    state = self.state[p]
                    state["step"] = 0
                    state["exp_avg_sq"] = torch.zeros_like(_neuron_norm(p))
                    scale = _neuron_norm(p).mean()
                    state["scale"] = scale if scale.item() != 0.0 else torch.tensor(0.01)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr, beta, constraints = group["lr"], group["beta"], group["constraints"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                state["step"] += 1
                bias_correction = 1 - beta ** state["step"]
                state["exp_avg_sq"] = beta * state["exp_avg_sq"] + (1 - beta) * _neuron_norm(p.grad) ** 2

                grad_normed = p.grad / (state["exp_avg_sq"] / bias_correction).sqrt()
                grad_normed = torch.nan_to_num(grad_normed, nan=0.0)

                p.data -= lr * state["scale"] * grad_normed

                if constraints and p.dim() > 1:
                    p.data -= _neuron_mean(p)
                    p.data /= _neuron_norm(p)
        return loss


class Muon(Optimizer):
    """Param groups must set `use_muon=True` (2D hidden-layer matrices) or
    `use_muon=False` (everything else) — see build_optimizer() below for how
    this model's parameters are split.
    """

    def __init__(self, params: Iterable[dict], lr: float = 0.02, momentum: float = 0.95,
                 ns_steps: int = 5, adamw_lr: float = 1e-3,
                 adamw_betas: tuple = (0.9, 0.95), adamw_eps: float = 1e-8):
        defaults = dict(lr=lr, momentum=momentum, ns_steps=ns_steps,
                         adamw_lr=adamw_lr, adamw_betas=adamw_betas, adamw_eps=adamw_eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            use_muon = group.get("use_muon", None)
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                is_muon_param = use_muon if use_muon is not None else (p.dim() == 2)

                if is_muon_param:
                    if "momentum_buf" not in state:
                        state["momentum_buf"] = torch.zeros_like(p)
                    buf = state["momentum_buf"]
                    buf.mul_(group["momentum"]).add_(g)
                    update = zeropower_via_newtonschulz(buf, steps=group["ns_steps"])
                    shape_scale = max(1.0, p.size(0) / p.size(1)) ** 0.5
                    p.add_(update, alpha=-group["lr"] * shape_scale)
                else:
                    if "step" not in state:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                    state["step"] += 1
                    beta1, beta2 = group["adamw_betas"]
                    state["exp_avg"].mul_(beta1).add_(g, alpha=1 - beta1)
                    state["exp_avg_sq"].mul_(beta2).addcmul_(g, g, value=1 - beta2)
                    bc1 = 1 - beta1 ** state["step"]
                    bc2 = 1 - beta2 ** state["step"]
                    denom = (state["exp_avg_sq"] / bc2).sqrt().add_(group["adamw_eps"])
                    step_size = group["adamw_lr"] / bc1
                    p.sub_(step_size * state["exp_avg"] / denom)
        return loss


def _muon_param_groups(model: torch.nn.Module, lr: float, adamw_lr: float) -> list[dict]:
    """Split params the way the Muon reference design expects: 2D weight
    matrices inside transformer blocks (attention/MLP) go through the
    orthogonalized-momentum branch; embeddings, output head, and norm/bias
    params go through the AdamW-style fallback branch.
    """
    hidden_matrices, other_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() == 2 and "blocks" in name:
            hidden_matrices.append(p)
        else:
            other_params.append(p)
    return [
        {"params": hidden_matrices, "use_muon": True, "lr": lr},
        {"params": other_params, "use_muon": False, "lr": lr, "adamw_lr": adamw_lr},
    ]


def build_optimizer(name: str, model: torch.nn.Module, lr: float, **kwargs) -> Optimizer:
    name = name.lower()
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr,
                                  betas=kwargs.get("betas", (0.9, 0.95)),
                                  weight_decay=kwargs.get("weight_decay", 0.0))
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr,
                                momentum=kwargs.get("momentum", 0.9))
    if name == "nero":
        return Nero(model.parameters(), lr=lr, beta=kwargs.get("beta", 0.999),
                     constraints=kwargs.get("constraints", True))
    if name == "muon":
        groups = _muon_param_groups(model, lr=lr, adamw_lr=kwargs.get("adamw_lr", lr))
        return Muon(groups, lr=lr, momentum=kwargs.get("momentum", 0.95))
    raise ValueError(f"unknown optimizer: {name}")
