from src.text_utils import exploratory_tokens, strip_html_breaks, tokenize, top_ngrams


def test_tokenize():
    assert tokenize("Hello, WORLD! It's good.") == ["hello", "world", "it's", "good"]


def test_strip_html_breaks():
    assert strip_html_breaks("one<br />two<br/>three<br>four") == "one two three four"


def test_exploratory_tokens_drop_stopwords_and_br():
    text = "This is a great movie<br />and the plot is bad."
    assert exploratory_tokens(text) == ["great", "movie", "plot", "bad"]
    assert exploratory_tokens(text, drop_stopwords=False) == [
        "this", "is", "a", "great", "movie", "and", "the", "plot", "is", "bad",
    ]


def test_bigrams_do_not_cross_documents():
    result = dict(top_ngrams(["very good", "very bad"], n=2, drop_stopwords=False))
    assert result == {"very good": 1, "very bad": 1}


def test_bigrams_skip_stopwords_within_document():
    result = dict(top_ngrams(["this movie is very good"], n=2))
    assert result == {"movie good": 1}
