from src.text_utils import content_tokens, log_odds, strip_html, tokenize, top_ngrams

def test_tokenize():
    assert tokenize("Hello, WORLD! It's good.") == ["hello", "world", "it's", "good"]

def test_bigrams_do_not_cross_documents():
    result = dict(top_ngrams(["very good", "very bad"], n=2))
    assert result == {"very good": 1, "very bad": 1}

def test_strip_html_removes_br_tags():
    assert tokenize(strip_html("good<br /><br />bad")) == ["good", "bad"]

def test_content_tokens_drop_noise_but_keep_negations():
    assert content_tokens("This movie is not good<br />at all!") == ["not", "good"]

def test_filtered_bigrams_keep_negation_and_skip_function_words():
    result = dict(top_ngrams(["it is not good", "of the not good"], n=2, skip_stopwords=True))
    assert result == {"not good": 2}

def test_log_odds_ranks_class_specific_words():
    pos, neg = log_odds(["great fun"] * 6, ["awful boring"] * 6, min_count=5, top_k=2)
    assert {w for w, _, _ in pos} == {"great", "fun"}
    assert {w for w, _, _ in neg} == {"awful", "boring"}
