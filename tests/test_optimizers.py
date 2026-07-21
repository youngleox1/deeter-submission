import torch

from src.model import DecoderOnlyTransformer, ModelConfig
from src.optimizers import (
    Lamb,
    Nero,
    build_optimizer,
    zeropower_per_head,
    zeropower_via_newtonschulz,
)


def _tiny_cfg() -> ModelConfig:
    return ModelConfig(
        vocab_size=17, max_seq_len=8,
        d_model=16, n_layers=2, n_heads=2, mlp_ratio=2, dropout=0.0,
    )


def _tiny_batch(cfg: ModelConfig, batch_size=4):
    idx = torch.randint(0, cfg.vocab_size, (batch_size, cfg.max_seq_len))
    targets = torch.randint(0, cfg.vocab_size, (batch_size, cfg.max_seq_len))
    return idx, targets


def test_newtonschulz_bounds_singular_values_near_one():
    """The quintic Newton-Schulz iteration used by Muon is tuned to bound
    singular values into a range around 1 (empirically ~[0.6, 1.3] for a
    handful of steps) -- it is NOT designed to converge to exact
    orthogonality, and running more steps does not tighten this further
    (the polynomial has this band as its fixed-point behavior). This test
    checks that real, documented behavior rather than exact orthogonality,
    which would be the wrong invariant to assert here.
    """
    torch.manual_seed(0)
    G = torch.randn(8, 5)  # tall matrix: 8 rows, 5 cols
    input_svals = torch.linalg.svdvals(G)
    assert input_svals.max() > 2.0  # input is nowhere near orthogonal

    X = zeropower_via_newtonschulz(G, steps=5)
    output_svals = torch.linalg.svdvals(X)
    assert output_svals.min() > 0.5
    assert output_svals.max() < 1.5


def test_nero_centers_and_projects_to_unit_norm_at_construction():
    """Matches the reference implementation (jxbz/nero): a >1D parameter is
    immediately mean-centered and projected to unit norm per neuron (row)
    when the optimizer is constructed, before any step is taken.
    """
    torch.manual_seed(0)
    weight = torch.nn.Parameter(torch.randn(6, 4) * 5 + 3)  # arbitrary scale/offset
    Nero([weight], lr=0.5)

    row_norms = weight.detach().norm(dim=1)
    row_means = weight.detach().mean(dim=1)
    assert torch.allclose(row_norms, torch.ones(6), atol=1e-5)
    assert torch.allclose(row_means, torch.zeros(6), atol=1e-5)


def test_nero_projection_invariant_preserved_after_step():
    """The centered/unit-norm invariant established at construction must
    still hold after a step -- the reference re-centers and re-projects
    to unit norm every step, not just once at init.
    """
    torch.manual_seed(0)
    weight = torch.nn.Parameter(torch.randn(6, 4))
    opt = Nero([weight], lr=0.5)

    weight.grad = torch.randn_like(weight) * 10  # large grad to stress-test projection
    opt.step()

    row_norms = weight.detach().norm(dim=1)
    row_means = weight.detach().mean(dim=1)
    assert torch.allclose(row_norms, torch.ones(6), atol=1e-4)
    assert torch.allclose(row_means, torch.zeros(6), atol=1e-4)


def test_nero_1d_params_are_not_centered_but_still_updated():
    """1D params (e.g. biases) can't be centered (a size-1 'neuron' has no
    meaningful mean-zero projection) -- the reference explicitly disallows
    this and just runs the same per-"neuron" (here, per-scalar) adaptive
    update uncentered.
    """
    torch.manual_seed(0)
    bias = torch.nn.Parameter(torch.randn(5) + 2.0)
    original = bias.detach().clone()
    opt = Nero([bias], lr=0.1)

    bias.grad = torch.randn_like(bias)
    opt.step()

    assert not torch.allclose(bias.detach(), original)  # it did update
    # no norm=1 or mean=0 invariant is expected/enforced for 1D params


def test_nero_has_no_momentum_state():
    """Nero is momentum-free by design (Liu et al., ICML'21) -- only a
    second-moment (RMS) running average is tracked, never a first-moment
    gradient EMA. An earlier version of this optimizer incorrectly added
    momentum to its 1D-parameter fallback branch; this guards against
    that regression.
    """
    torch.manual_seed(0)
    weight = torch.nn.Parameter(torch.randn(6, 4))
    bias = torch.nn.Parameter(torch.randn(4))
    opt = Nero([weight, bias], lr=0.1)

    weight.grad = torch.randn_like(weight)
    bias.grad = torch.randn_like(bias)
    opt.step()

    for p in (weight, bias):
        state = opt.state[p]
        assert "exp_avg" not in state
        assert "exp_avg_sq" in state


def test_muon_param_groups_split_hidden_vs_other():
    cfg = _tiny_cfg()
    model = DecoderOnlyTransformer(cfg)
    opt = build_optimizer("muon", model, lr=0.02)

    muon_group = next(g for g in opt.param_groups if g["use_muon"])
    other_group = next(g for g in opt.param_groups if not g["use_muon"])

    assert all(p.dim() == 2 for p in muon_group["params"])
    assert len(muon_group["params"]) > 0
    assert len(other_group["params"]) > 0
    # tok_emb / pos_emb / head must NOT be in the Muon (hidden-matrix) branch
    excluded = {model.tok_emb.weight, model.pos_emb.weight, model.head.weight}
    muon_param_ids = {id(p) for p in muon_group["params"]}
    assert all(id(p) not in muon_param_ids for p in excluded)


def test_all_optimizers_decrease_loss_on_tiny_model():
    cfg = _tiny_cfg()
    torch.manual_seed(0)
    idx, targets = _tiny_batch(cfg)

    for name, lr in [("adamw", 3e-3), ("sgd", 1e-2), ("nero", 1e-2), ("muon", 2e-2),
                     ("lamb", 3e-3), ("muon_per_head", 2e-2)]:
        torch.manual_seed(1)
        model = DecoderOnlyTransformer(cfg)
        opt = build_optimizer(name, model, lr=lr)

        _, initial_loss = model(idx, targets)
        for _ in range(50):
            opt.zero_grad()
            _, loss = model(idx, targets)
            loss.backward()
            opt.step()
        _, final_loss = model(idx, targets)

        assert torch.isfinite(final_loss), f"{name} diverged"
        assert final_loss.item() < initial_loss.item(), f"{name} did not reduce loss"


def test_lamb_trust_ratio_defaults_to_one_when_param_is_zero():
    """Reference behavior: if the parameter's own norm is exactly zero
    (weight_norm == 0), trust_ratio falls back to 1.0 rather than
    dividing by (near-)zero or collapsing the update to nothing.
    """
    torch.manual_seed(0)
    beta1, beta2, eps, lr = 0.9, 0.999, 1e-6, 0.1
    param = torch.nn.Parameter(torch.zeros(4, 4))
    opt = Lamb([param], lr=lr, betas=(beta1, beta2), eps=eps)
    grad = torch.ones_like(param)
    param.grad = grad.clone()
    opt.step()

    # with trust_ratio=1.0 (since param started at exactly zero norm):
    # update = -lr * adam_step, no bias correction, so
    # adam_step = (1-beta1)*grad / (sqrt((1-beta2)*grad^2) + eps)
    exp_avg = (1 - beta1) * grad
    exp_avg_sq = (1 - beta2) * grad ** 2
    adam_step = exp_avg / (exp_avg_sq.sqrt() + eps)
    expected = -lr * 1.0 * adam_step  # trust_ratio forced to 1.0 by the zero-norm fallback

    assert torch.allclose(param.detach(), expected, atol=1e-5)


def test_lamb_no_bias_correction():
    """Reference implementation explicitly omits bias correction (a real,
    easy-to-miss deviation from naive Adam intuition) -- checked by
    comparing against a hand-computed first-step update with no bias
    correction terms.
    """
    torch.manual_seed(0)
    param = torch.nn.Parameter(torch.tensor([1.0, 1.0, 1.0, 1.0]))
    grad = torch.tensor([2.0, 2.0, 2.0, 2.0])
    lr, beta1, beta2, eps = 0.1, 0.9, 0.999, 1e-6

    opt = Lamb([param], lr=lr, betas=(beta1, beta2), eps=eps)
    param.grad = grad.clone()
    opt.step()

    exp_avg = (1 - beta1) * grad
    exp_avg_sq = (1 - beta2) * grad ** 2
    adam_step = exp_avg / (exp_avg_sq.sqrt() + eps)  # NO bias correction division
    weight_norm = torch.tensor([1.0, 1.0, 1.0, 1.0]).norm().clamp(0, 10)  # pre-update param norm
    adam_norm = adam_step.norm()
    trust_ratio = (weight_norm / adam_norm).item()
    expected = torch.ones(4) - lr * trust_ratio * adam_step

    assert torch.allclose(param.detach(), expected, atol=1e-4)


def test_zeropower_per_head_isolates_heads():
    """The defining correctness property of 'per-head': changing one
    head's input block must not change another head's output block.
    Without this, it would just be a reshaped whole-matrix operation.
    """
    torch.manual_seed(0)
    n_heads = 4
    G = torch.randn(n_heads * 6, 10)  # 4 heads x head_dim=6, in_dim=10

    out_a = zeropower_per_head(G, n_heads=n_heads, steps=5)

    G_modified = G.clone()
    G_modified[0:6] += 100.0  # drastically change only head 0's block

    out_b = zeropower_per_head(G_modified, n_heads=n_heads, steps=5)

    # head 0's output changed, but heads 1-3 must be untouched
    assert not torch.allclose(out_a[0:6], out_b[0:6])
    assert torch.allclose(out_a[6:], out_b[6:], atol=1e-6)


def test_muon_per_head_param_groups_shapes():
    cfg = _tiny_cfg()
    model = DecoderOnlyTransformer(cfg)
    opt = build_optimizer("muon_per_head", model, lr=0.02)

    qkv_group = next(g for g in opt.param_groups if g.get("per_head") and g["n_heads"] == 3 * cfg.n_heads)
    proj_group = next(g for g in opt.param_groups if g.get("per_head") and g["n_heads"] == cfg.n_heads)
    mlp_group = next(g for g in opt.param_groups if g["use_muon"] and not g.get("per_head", False))
    other_group = next(g for g in opt.param_groups if not g["use_muon"])

    assert qkv_group["head_axis"] == 0
    assert proj_group["head_axis"] == 1
    assert len(qkv_group["params"]) > 0
    assert len(proj_group["params"]) > 0
    assert len(mlp_group["params"]) > 0
    assert len(other_group["params"]) > 0
    # every qkv weight's row count must actually be divisible by 3*n_heads
    for p in qkv_group["params"]:
        assert p.size(0) % (3 * cfg.n_heads) == 0
    for p in proj_group["params"]:
        assert p.size(1) % cfg.n_heads == 0
