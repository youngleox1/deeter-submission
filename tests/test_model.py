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
