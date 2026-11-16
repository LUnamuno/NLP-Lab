from src.run import bootstrap_macro_f1, chunk_token_ids, transformer_eval_size


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


def test_transformer_eval_size():
    assert transformer_eval_size(0, 1000) == 0
    assert transformer_eval_size(-1, 1000) == 1000
    assert transformer_eval_size(40, 1000) == 40
    assert transformer_eval_size(4000, 1000) == 1000


def test_chunk_token_ids_keeps_short_sequence():
    assert chunk_token_ids([1, 2, 3], content_max=8) == [[1, 2, 3]]


def test_chunk_token_ids_covers_all_tokens_without_overlap():
    ids = list(range(10))
    chunks = chunk_token_ids(ids, content_max=4, stride=4)
    assert chunks == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]
    assert [tok for chunk in chunks for tok in chunk] == ids


def test_chunk_token_ids_overlapping_windows_still_reach_the_end():
    ids = list(range(10))
    chunks = chunk_token_ids(ids, content_max=6, stride=3)
    assert chunks[0] == [0, 1, 2, 3, 4, 5]
    assert chunks[-1][-1] == 9
    assert all(len(chunk) <= 6 for chunk in chunks)
