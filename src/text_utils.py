import re
from collections import Counter

TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")

def tokenize(text: str) -> list[str]:
    """Lowercase word tokenization for exploratory analysis only."""
    return TOKEN_RE.findall(text.lower())


def top_ngrams(texts: list[str], n: int = 2, top_k: int = 20) -> list[tuple[str, int]]:
    counts: Counter[str] = Counter()
    for text in texts:
        tokens = tokenize(text)
        counts.update(" ".join(tokens[i:i+n]) for i in range(len(tokens)-n+1))
    return counts.most_common(top_k)
