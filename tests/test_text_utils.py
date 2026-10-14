from src.text_utils import tokenize, top_ngrams

def test_tokenize():
    assert tokenize("Hello, WORLD! It's good.") == ["hello", "world", "it's", "good"]

def test_bigrams_do_not_cross_documents():
    result = dict(top_ngrams(["very good", "very bad"], n=2))
    assert result == {"very good": 1, "very bad": 1}
