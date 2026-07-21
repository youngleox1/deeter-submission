"""Optimizer candidates under comparison: AdamW, SGD+momentum, Nero, Muon.

AdamW and SGD are used directly from torch.optim — no reimplementation
needed or wanted. Nero and Muon are implemented from scratch below.

Both custom implementations are simplified reproductions of the published
methods, not verbatim ports of reference code, and are documented as such:

- Nero (Liu, Bernstein, Meister, Yue. "Learning by Turning: Neural
  Architecture-Aware Optimisation." ICML 2021): neuron-wise (per-output-row)
  normalized gradient step with a running second-moment estimate, followed
  by a projection that restores each neuron's pre-update row norm ("sphere
  projection" — since a neuron's overall scale is often absorbed by a
  downstream normalization layer, constraining it lets the optimizer focus
  purely on direction). 1D parameters (biases, norm affine params) don't
  have a natural row/neuron structure, so they fall back to a plain
  elementwise normalized (Adam-like) update — a simplification relative to
  the original paper, noted here rather than left implicit.

- Muon (Jordan et al., "Muon: An optimizer for hidden layers in neural
  networks," 2024 — technical report, not peer-reviewed): momentum, then
  approximate orthogonalization of the momentum matrix via Newton-Schulz
  iteration, applied only to 2D hidden-layer weight matrices. Following the
  reference design, this is a hybrid optimizer: parameters flagged
  `use_muon=False` in their param group (embeddings, output head, biases,
  norm params) instead receive a plain AdamW-style update in the same
  optimizer step.
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


class Nero(Optimizer):
    def __init__(self, params: Iterable[torch.nn.Parameter], lr: float = 0.01,
                 beta: float = 0.999, eps: float = 1e-8):
        defaults = dict(lr=lr, beta=beta, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr, beta, eps = group["lr"], group["beta"], group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    if p.dim() >= 2:
                        p_flat0 = p.view(p.size(0), -1)
                        state["v"] = torch.zeros(p.size(0), device=p.device, dtype=p.dtype)
                        state["init_row_norm"] = p_flat0.norm(dim=1).clamp_min(eps)
                    else:
                        state["v"] = torch.zeros_like(p)
                state["step"] += 1

                if p.dim() >= 2:
                    g_flat = g.view(g.size(0), -1)
                    row_sq = g_flat.pow(2).mean(dim=1)
                    state["v"].mul_(beta).add_(row_sq, alpha=1 - beta)
                    bias_corr = 1 - beta ** state["step"]
                    denom = (state["v"] / bias_corr).sqrt().add_(eps)
                    update = g_flat / denom.unsqueeze(1)

                    p_flat = p.view(p.size(0), -1)
                    p_flat.add_(update, alpha=-lr)

                    cur_norm = p_flat.norm(dim=1).clamp_min(eps)
                    scale = state["init_row_norm"] / cur_norm
                    p_flat.mul_(scale.unsqueeze(1))
                else:
                    state["v"].mul_(beta).addcmul_(g, g, value=1 - beta)
                    bias_corr = 1 - beta ** state["step"]
                    denom = (state["v"] / bias_corr).sqrt().add_(eps)
                    p.add_(g / denom, alpha=-lr)
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
        return Nero(model.parameters(), lr=lr, beta=kwargs.get("beta", 0.999))
    if name == "muon":
        groups = _muon_param_groups(model, lr=lr, adamw_lr=kwargs.get("adamw_lr", lr))
        return Muon(groups, lr=lr, momentum=kwargs.get("momentum", 0.95))
    raise ValueError(f"unknown optimizer: {name}")
