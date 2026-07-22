"""Small decoder-only transformer, shared by the core (text) and stretch
(finance) experiments. Architecture is deliberately standard (pre-norm,
causal self-attention) so the optimizer comparison isn't confounded by
architectural novelty.
"""
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int
    max_seq_len: int
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    mlp_ratio: int = 4
    dropout: float = 0.0
    layernorm_affine: bool = True
    continuous_input: bool = False  # finance v3: linear-project a real-valued
    # input (e.g. vol-scaled returns) instead of a token-embedding lookup,
    # while the output head still classifies into discrete vocab_size bins
    # (see forward()) -- isolates whether it's the objective or the input
    # representation that matters more for the finance stretch's input-
    # binarization confound (see README, finance v2 section).


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        qkv = self.qkv(x).view(b, t, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = (a.transpose(1, 2) for a in (q, k, v))  # (b, h, t, hd)
        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).reshape(b, t, c)
        return self.proj(out)


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        hidden = cfg.mlp_ratio * cfg.d_model
        self.fc1 = nn.Linear(cfg.d_model, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model, elementwise_affine=cfg.layernorm_affine)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model, elementwise_affine=cfg.layernorm_affine)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class DecoderOnlyTransformer(nn.Module):
    """Sequence model used for both experiments.

    Core (text): outputs next-token logits over a vocabulary, discrete
    token-id input via embedding lookup.
    Finance stretch: same trunk and output head (still classifies into
    discrete vocab_size bins). Input is either discrete token ids (v1/v2,
    `continuous_input=False`, embedding lookup) or real-valued (v3,
    `continuous_input=True`, linear projection of a raw scalar per
    position) -- see ModelConfig.continuous_input.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.continuous_input:
            self.input_proj = nn.Linear(1, cfg.d_model)
        else:
            self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model, elementwise_affine=cfg.layernorm_affine)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        b, t = idx.shape
        assert t <= self.cfg.max_seq_len, "sequence longer than max_seq_len"
        pos = torch.arange(t, device=idx.device)
        if self.cfg.continuous_input:
            x = self.input_proj(idx.float().unsqueeze(-1)) + self.pos_emb(pos)[None, :, :]
        else:
            x = self.tok_emb(idx) + self.pos_emb(pos)[None, :, :]
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )
        return logits, loss
