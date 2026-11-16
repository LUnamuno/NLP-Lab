import re
from collections import Counter

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")
BR_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
# Exploratory visuals only: classifiers keep raw text (negation matters).
STOPWORDS = frozenset(ENGLISH_STOP_WORDS) | {"br"}


def strip_html_breaks(text: str) -> str:
    """Replace leftover IMDb HTML line breaks with spaces."""
    return BR_RE.sub(" ", text)


def tokenize(text: str) -> list[str]:
    """Lowercase word tokenization for exploratory analysis only."""
    return TOKEN_RE.findall(text.lower())


def exploratory_tokens(text: str, *, drop_stopwords: bool = True) -> list[str]:
    """Tokens for frequency tables / word clouds: strip <br /> and optionally stopwords."""
    tokens = tokenize(strip_html_breaks(text))
    if drop_stopwords:
        tokens = [tok for tok in tokens if tok not in STOPWORDS]
    return tokens


def top_ngrams(
    texts: list[str],
    n: int = 2,
    top_k: int = 20,
    *,
    drop_stopwords: bool = True,
) -> list[tuple[str, int]]:
    counts: Counter[str] = Counter()
    for text in texts:
        tokens = exploratory_tokens(text, drop_stopwords=drop_stopwords)
        counts.update(" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1))
    return counts.most_common(top_k)
