import math
import re
from collections import Counter

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")
HTML_TAG_RE = re.compile(r"<[^>]+>")

# Negations carry sentiment even though scikit-learn lists them as stopwords.
NEGATIONS = {"no", "not", "nor", "never", "nothing", "nobody", "none", "cannot"}
# "movie"/"film" are domain stopwords: they are frequent in both classes.
STOPWORDS = (frozenset(ENGLISH_STOP_WORDS) - NEGATIONS) | {"movie", "film", "movies", "films", "br"}


def strip_html(text: str) -> str:
    """Replace HTML tags (IMDb reviews contain `<br />`) with spaces."""
    return HTML_TAG_RE.sub(" ", text)


def tokenize(text: str) -> list[str]:
    """Lowercase word tokenization for exploratory analysis only."""
    return TOKEN_RE.findall(text.lower())


def content_tokens(text: str) -> list[str]:
    """Tokens without HTML, stopwords or one-letter fragments (exploration only)."""
    return [t for t in tokenize(strip_html(text)) if t not in STOPWORDS and len(t) > 1]


def top_ngrams(
    texts: list[str], n: int = 2, top_k: int = 20, skip_stopwords: bool = False
) -> list[tuple[str, int]]:
    """Most frequent n-grams. N-grams never cross document boundaries.

    With ``skip_stopwords`` HTML is removed and n-grams containing any stopword
    are dropped (negations are kept, so "not good" survives).
    """
    counts: Counter[str] = Counter()
    for text in texts:
        tokens = tokenize(strip_html(text) if skip_stopwords else text)
        for i in range(len(tokens) - n + 1):
            gram = tokens[i:i + n]
            if skip_stopwords and any(t in STOPWORDS for t in gram):
                continue
            counts[" ".join(gram)] += 1
    return counts.most_common(top_k)


def log_odds(
    pos_texts: list[str], neg_texts: list[str], min_count: int = 5, top_k: int = 15
) -> tuple[list[tuple[str, float, int]], list[tuple[str, float, int]]]:
    """Words most associated with the positive / negative class.

    Smoothed log-odds ratio over content tokens, with (word, score, total_count);
    words rarer than ``min_count`` are ignored because their ratio is noise.
    """
    pos = Counter(t for text in pos_texts for t in content_tokens(text))
    neg = Counter(t for text in neg_texts for t in content_tokens(text))
    n_pos, n_neg = sum(pos.values()), sum(neg.values())
    scored = []
    for word in set(pos) | set(neg):
        total = pos[word] + neg[word]
        if total < min_count:
            continue
        p = (pos[word] + 1) / (n_pos + 1)
        q = (neg[word] + 1) / (n_neg + 1)
        scored.append((word, math.log(p / (1 - p)) - math.log(q / (1 - q)), total))
    scored.sort(key=lambda row: row[1])
    return scored[::-1][:top_k], scored[:top_k]
