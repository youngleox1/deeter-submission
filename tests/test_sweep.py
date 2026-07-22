import csv

import yaml

from src.sweep import CSV_FIELDS, run_sweep, save_results_csv


def _tiny_sweep_config():
    return {
        "data": {"type": "text"},
        "model": {
            "max_seq_len": 16, "d_model": 16, "n_layers": 2,
            "n_heads": 2, "mlp_ratio": 2, "dropout": 0.0,
        },
        "train": {
            "batch_size": 8, "seq_len": 16, "max_steps": 10,
            "eval_interval": 10, "eval_iters": 2,
        },
        "sweep": {
            "seeds": [0, 1],
            "lr_grids": {
                "adamw": [1e-3, 3e-3],
                "sgd": [1e-2, 3e-2],
            },
            "optimizer_kwargs": {},
        },
    }


def test_run_sweep_produces_expected_number_of_results():
    results = run_sweep(_tiny_sweep_config())
    # 2 optimizers x 2 lrs x 2 seeds = 8 runs
    assert len(results) == 8
    assert {r["optimizer_name"] for r in results} == {"adamw", "sgd"}
    assert all("best_val_loss" in r for r in results)


def test_save_results_csv_writes_all_rows_with_expected_columns(tmp_path):
    results = run_sweep(_tiny_sweep_config())
    out_path = tmp_path / "results" / "sweep.csv"
    save_results_csv(results, out_path)

    with open(out_path, newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == len(results)
    assert set(rows[0].keys()) == set(CSV_FIELDS)


def _assert_sweep_config_structure(path, expected_optimizers):
    with open(path) as f:
        config = yaml.safe_load(f)

    assert config["data"]["type"] in ("text", "finance")
    assert "sweep" in config and "lr_grids" in config["sweep"]
    lr_grids = config["sweep"]["lr_grids"]
    assert set(lr_grids.keys()) == expected_optimizers
    for name, lrs in lr_grids.items():
        assert len(lrs) == 9, f"{name} grid should have 9 points"
        assert lrs == sorted(lrs), f"{name} grid should be sorted ascending"


def test_core_sweep_yaml_has_expected_structure():
    _assert_sweep_config_structure("configs/core_sweep.yaml", {"adamw", "sgd", "nero", "muon"})


def test_finance_sweep_yaml_has_expected_structure():
    _assert_sweep_config_structure("configs/finance_sweep.yaml", {"adamw", "sgd", "nero", "muon"})


def test_finance_v2_sweep_yaml_has_expected_structure():
    _assert_sweep_config_structure("configs/finance_v2_sweep.yaml", {"adamw", "muon"})


def test_finance_v3_sweep_yaml_has_expected_structure_and_continuous_input():
    _assert_sweep_config_structure("configs/finance_v3_sweep.yaml", {"adamw", "muon"})
    with open("configs/finance_v3_sweep.yaml") as f:
        config = yaml.safe_load(f)
    assert config["data"]["continuous_input"] is True
    assert config["model"]["continuous_input"] is True
