import torch

from src.model import DecoderOnlyTransformer, ModelConfig


def _tiny_cfg(vocab_size=17, max_seq_len=8) -> ModelConfig:
    return ModelConfig(
        vocab_size=vocab_size, max_seq_len=max_seq_len,
        d_model=16, n_layers=2, n_heads=2, mlp_ratio=2, dropout=0.0,
    )


def test_forward_shape():
    cfg = _tiny_cfg()
    model = DecoderOnlyTransformer(cfg)
    idx = torch.randint(0, cfg.vocab_size, (3, cfg.max_seq_len))
    logits, loss = model(idx)
    assert logits.shape == (3, cfg.max_seq_len, cfg.vocab_size)
    assert loss is None


def test_loss_computed_when_targets_given():
    cfg = _tiny_cfg()
    model = DecoderOnlyTransformer(cfg)
    idx = torch.randint(0, cfg.vocab_size, (3, cfg.max_seq_len))
    targets = torch.randint(0, cfg.vocab_size, (3, cfg.max_seq_len))
    _, loss = model(idx, targets)
    assert loss is not None
    assert loss.item() > 0
    assert torch.isfinite(loss)


def test_causal_masking_no_future_leakage():
    """Changing a future token must not change logits at earlier positions.

    This is the correctness property the whole experiment leans on: without
    it, validation loss wouldn't measure genuine next-token prediction.
    """
    cfg = _tiny_cfg()
    model = DecoderOnlyTransformer(cfg)
    model.eval()

    idx = torch.randint(0, cfg.vocab_size, (1, cfg.max_seq_len))
    idx_modified = idx.clone()
    last_pos = cfg.max_seq_len - 1
    # force the last token to a different value
    idx_modified[0, last_pos] = (idx[0, last_pos] + 1) % cfg.vocab_size

    with torch.no_grad():
        logits, _ = model(idx)
        logits_modified, _ = model(idx_modified)

    # all positions strictly before the changed one must be identical
    assert torch.allclose(
        logits[:, :last_pos], logits_modified[:, :last_pos], atol=1e-6
    )
    # the changed position itself is allowed (expected) to differ
    assert not torch.allclose(
        logits[:, last_pos], logits_modified[:, last_pos], atol=1e-6
    )


def test_backward_pass_runs():
    cfg = _tiny_cfg()
    model = DecoderOnlyTransformer(cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, cfg.max_seq_len))
    targets = torch.randint(0, cfg.vocab_size, (2, cfg.max_seq_len))
    _, loss = model(idx, targets)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None for g in grads)
    assert all(torch.isfinite(g).all() for g in grads)


def test_layernorm_affine_false_removes_ln_params_and_still_runs():
    """Ablation switch for the Nero hypothesis: Nero's sphere projection
    assumes a neuron's scale is irrelevant because downstream normalization
    absorbs it -- which isn't quite true if that normalization layer has
    its own learnable affine scale/shift. This just checks the switch
    actually removes those params and the model still runs.
    """
    cfg = _tiny_cfg()
    cfg.layernorm_affine = False
    model = DecoderOnlyTransformer(cfg)

    for module in model.modules():
        if isinstance(module, torch.nn.LayerNorm):
            assert module.weight is None
            assert module.bias is None

    idx = torch.randint(0, cfg.vocab_size, (2, cfg.max_seq_len))
    targets = torch.randint(0, cfg.vocab_size, (2, cfg.max_seq_len))
    _, loss = model(idx, targets)
    assert torch.isfinite(loss)
    loss.backward()


def test_continuous_input_uses_linear_projection_not_embedding():
    """Finance v3: real-valued input (e.g. vol-scaled returns) via a
    linear projection, while the output head still classifies into
    discrete vocab_size bins -- only the INPUT side changes.
    """
    cfg = _tiny_cfg()
    cfg.continuous_input = True
    model = DecoderOnlyTransformer(cfg)

    assert not hasattr(model, "tok_emb")
    assert hasattr(model, "input_proj")
    assert isinstance(model.input_proj, torch.nn.Linear)
    assert model.input_proj.in_features == 1
    assert model.input_proj.out_features == cfg.d_model
    # output head is unchanged: still a discrete classifier over vocab_size
    assert model.head.out_features == cfg.vocab_size


def test_continuous_input_forward_and_backward_run():
    cfg = _tiny_cfg()
    cfg.continuous_input = True
    model = DecoderOnlyTransformer(cfg)

    idx = torch.randn(3, cfg.max_seq_len)  # real-valued, NOT token ids
    targets = torch.randint(0, cfg.vocab_size, (3, cfg.max_seq_len))  # still discrete
    logits, loss = model(idx, targets)

    assert logits.shape == (3, cfg.max_seq_len, cfg.vocab_size)
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None for g in grads)
    assert all(torch.isfinite(g).all() for g in grads)


def test_continuous_input_different_values_give_different_logits():
    """Sanity: the linear projection must actually be sensitive to the
    input's real value, not silently ignoring it (e.g. via a shape bug
    that broadcasts away the real content).
    """
    cfg = _tiny_cfg()
    cfg.continuous_input = True
    model = DecoderOnlyTransformer(cfg)
    model.eval()

    idx_a = torch.zeros(1, cfg.max_seq_len)
    idx_b = torch.ones(1, cfg.max_seq_len) * 5.0

    with torch.no_grad():
        logits_a, _ = model(idx_a)
        logits_b, _ = model(idx_b)

    assert not torch.allclose(logits_a, logits_b)
