import numpy as np
import torch

from src.eval_finance import evaluate_directional_metrics


class _FakeTokenizer:
    def __init__(self, bin_mean_return):
        self.bin_mean_return = bin_mean_return


class _FakeData:
    """Returns a fixed, hand-picked (x, y) batch regardless of args, so the
    expected accuracy/Brier numbers can be computed by hand.
    """
    def __init__(self, x, y, bin_mean_return, continuous_input=False, majority_direction_train=1.0):
        self._x, self._y = x, y
        self.tokenizer = _FakeTokenizer(bin_mean_return)
        self.continuous_input = continuous_input
        self.majority_direction_train = majority_direction_train

    def get_batch(self, split, batch_size, seq_len, device="cpu"):
        return self._x, self._y


class _FakeModel:
    """Always predicts a fixed bin via a near-one-hot logit vector."""
    def __init__(self, always_predict_bin, vocab_size):
        self.always_predict_bin = always_predict_bin
        self.vocab_size = vocab_size

    def eval(self):
        pass

    def __call__(self, x, y):
        b, t = x.shape
        logits = torch.full((b, t, self.vocab_size), -10.0)
        logits[..., self.always_predict_bin] = 10.0
        return logits, None


def test_directional_accuracy_and_brier_hand_computed_case():
    # bins 0,1 -> negative direction; bins 2,3 -> positive direction
    bin_mean_return = np.array([-0.02, -0.01, 0.01, 0.02])
    x = torch.tensor([[0, 2]])  # yesterday: neg, pos
    y = torch.tensor([[2, 2]])  # today (actual): pos, pos

    data = _FakeData(x, y, bin_mean_return, majority_direction_train=1.0)
    model = _FakeModel(always_predict_bin=2, vocab_size=4)  # model always predicts "pos"

    result = evaluate_directional_metrics(
        model, data, batch_size=1, seq_len=2, n_eval_batches=1, device="cpu"
    )

    # persistence (predict yesterday's direction): [neg vs pos -> wrong, pos vs pos -> correct] = 0.5
    assert abs(result["persistence_directional_accuracy"] - 0.5) < 1e-9
    # majority=pos (train), actual=[pos,pos] -> both correct = 1.0
    assert abs(result["majority_directional_accuracy"] - 1.0) < 1e-9
    # model always predicts pos, actual is [pos, pos] -> both correct
    assert result["model_directional_accuracy"] == 1.0
    # model is near-certain and correct both times -> Brier near 0
    assert result["brier_score"] < 0.01
    assert result["n_predictions"] == 2


def test_majority_baseline_uses_train_majority_not_val_data():
    """The majority baseline is a single fixed value (data.majority_direction_train),
    computed from TRAIN data at FinanceReturns construction time -- this test
    just confirms evaluate_directional_metrics actually uses that stored
    value rather than deriving majority from the val batch itself (which
    would leak val-period information into the baseline).
    """
    bin_mean_return = np.array([-0.02, -0.01, 0.01, 0.02])
    x = torch.tensor([[0, 0]])
    y = torch.tensor([[0, 0]])  # actual is all-negative in this val batch

    # majority_direction_train says "positive" despite this val batch being
    # all-negative -- majority accuracy should reflect that mismatch (0.0),
    # not adapt to what the val batch happens to contain.
    data = _FakeData(x, y, bin_mean_return, majority_direction_train=1.0)
    model = _FakeModel(always_predict_bin=0, vocab_size=4)

    result = evaluate_directional_metrics(
        model, data, batch_size=1, seq_len=2, n_eval_batches=1, device="cpu"
    )
    assert abs(result["majority_directional_accuracy"] - 0.0) < 1e-9


def test_continuous_input_persistence_baseline_uses_sign_of_x_not_lookup():
    """Regression test: x is a real-valued return in continuous_input mode,
    not a bin id -- indexing direction_lookup[x] (the discrete-mode path)
    would be wrong (or crash). Persistence direction must be sign(x) directly.
    """
    bin_mean_return = np.array([-0.02, -0.01, 0.01, 0.02])
    x = torch.tensor([[-0.015, 0.03]])  # continuous: negative, positive
    y = torch.tensor([[2, 2]])  # actual (discrete): pos, pos

    data = _FakeData(x, y, bin_mean_return, continuous_input=True)
    model = _FakeModel(always_predict_bin=2, vocab_size=4)

    result = evaluate_directional_metrics(
        model, data, batch_size=1, seq_len=2, n_eval_batches=1, device="cpu"
    )

    # persistence = sign(x) = [neg, pos]; actual = [pos, pos] -> [wrong, correct] = 0.5
    assert abs(result["persistence_directional_accuracy"] - 0.5) < 1e-9
