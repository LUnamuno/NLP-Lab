from src.run import bootstrap_macro_f1


def test_bootstrap_macro_f1_perfect_predictions_have_tight_ci():
    y = [0, 1, 0, 1] * 25
    out = bootstrap_macro_f1(y, y, n_boot=200, seed=0)
    assert out["macro_f1"] == 1.0
    assert out["macro_f1_ci95"] == [1.0, 1.0]


def test_bootstrap_macro_f1_ci_contains_point_estimate():
    y_true = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    y_pred = [0, 1, 0, 0, 0, 1, 1, 1, 0, 1]
    out = bootstrap_macro_f1(y_true, y_pred, n_boot=500, seed=42)
    low, high = out["macro_f1_ci95"]
    assert low <= out["macro_f1"] <= high
