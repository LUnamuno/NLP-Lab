"""Small, reproducible sentiment-analysis experiment on IMDb reviews."""
import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, f1_score
from sklearn.model_selection import train_test_split
from wordcloud import WordCloud
from .text_utils import content_tokens, log_odds, tokenize, top_ngrams

ROOT = Path(__file__).resolve().parents[1]
TRANSFORMER_NAME = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
LABELS = {0: "negative", 1: "positive"}


def load_reviews(n: int, seed: int) -> pd.DataFrame:
    # Download a modest balanced sample from the IMDb training split.
    ds = load_dataset("stanfordnlp/imdb", split="train")
    df = ds.to_pandas()[["text", "label"]]
    each = n // 2
    df = pd.concat([
        df.loc[df.label == label].sample(n=each, random_state=seed)
        for label in (0, 1)
    ], ignore_index=True)
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def explore(train: pd.DataFrame, out: Path) -> None:
    """Exploratory tables and plots. Uses training texts only, never feeds the models."""
    by_class = {name: train.loc[train.label == label, "text"].tolist() for label, name in LABELS.items()}
    for name, texts in by_class.items():
        raw = Counter(tok for text in texts for tok in tokenize(text))
        pd.DataFrame(raw.most_common(30), columns=["word", "frequency"]).to_csv(out / f"words_raw_{name}.csv", index=False)

        content = Counter(tok for text in texts for tok in content_tokens(text))
        pd.DataFrame(content.most_common(30), columns=["word", "frequency"]).to_csv(out / f"words_{name}.csv", index=False)

        bigrams = top_ngrams(texts, n=2, top_k=30, skip_stopwords=True)
        pd.DataFrame(bigrams, columns=["bigram", "frequency"]).to_csv(out / f"bigrams_{name}.csv", index=False)

        wc = WordCloud(width=900, height=450, background_color="white").generate_from_frequencies(content)
        wc.to_file(str(out / f"wordcloud_{name}.png"))

    pos_words, neg_words = log_odds(by_class["positive"], by_class["negative"])
    rows = [("positive", *r) for r in pos_words] + [("negative", *r) for r in neg_words]
    pd.DataFrame(rows, columns=["class", "word", "log_odds", "count"]).to_csv(out / "distinctive_words.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 6))
    words = [w for w, _, _ in neg_words[::-1]] + [w for w, _, _ in pos_words[::-1]]
    scores = [s for _, s, _ in neg_words[::-1]] + [s for _, s, _ in pos_words[::-1]]
    ax.barh(words, scores, color=["#c0504d" if s < 0 else "#4f81bd" for s in scores])
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Log-odds ratio (positive vs. negative)")
    ax.set_title("Most distinctive words per class (training split)")
    fig.tight_layout()
    fig.savefig(out / "distinctive_words.png", dpi=150)
    plt.close(fig)


def bootstrap_f1_difference(y, pred_a, pred_b, seed: int, rounds: int = 2000) -> tuple[float, float]:
    """95% paired-bootstrap interval for macro-F1(a) - macro-F1(b)."""
    rng = np.random.default_rng(seed)
    y, pred_a, pred_b = map(np.asarray, (y, pred_a, pred_b))
    diffs = []
    for _ in range(rounds):
        idx = rng.integers(0, len(y), len(y))
        diffs.append(f1_score(y[idx], pred_a[idx], average="macro") - f1_score(y[idx], pred_b[idx], average="macro"))
    low, high = np.percentile(diffs, [2.5, 97.5])
    return float(low), float(high)


def main(n: int, seed: int, transformer_n: int, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    df = load_reviews(n, seed)
    train, test = train_test_split(df, test_size=0.25, stratify=df.label, random_state=seed)

    # Exploration is allowed to rewrite text (HTML, stopwords); the models below
    # always receive the raw review, since negation/punctuation matters.
    explore(train, out)

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=20000, strip_accents="unicode")
    X_train = vectorizer.fit_transform(train.text)
    X_test = vectorizer.transform(test.text)
    clf = LogisticRegression(max_iter=2000, random_state=seed)
    clf.fit(X_train, train.label)
    classical = clf.predict(X_test)
    report = {
        "config": {"n": n, "seed": seed, "train_size": len(train), "test_size": len(test)},
        "classical": classification_report(test.label, classical, output_dict=True, zero_division=0),
    }
    ConfusionMatrixDisplay.from_predictions(test.label, classical, display_labels=list(LABELS.values()))
    plt.tight_layout()
    plt.savefig(out / "confusion_matrix_classical.png", dpi=150)
    plt.close()

    predictions = test.assign(classical=classical)

    if transformer_n:
        from transformers import pipeline
        # This model is fine-tuned on SST-2, not IMDb: treat comparison as illustrative.
        model = pipeline("text-classification", model=TRANSFORMER_NAME, device=-1)
        sample = test.sample(n=min(transformer_n, len(test)), random_state=seed)
        # pipeline truncates long reviews at model max length.
        predicted = model(sample.text.tolist(), truncation=True, batch_size=8)
        transformer = [1 if item["label"].upper() == "POSITIVE" else 0 for item in predicted]
        predictions.loc[sample.index, "transformer"] = transformer
        predictions.loc[sample.index, "transformer_score"] = [item["score"] for item in predicted]
        classical_subset = clf.predict(vectorizer.transform(sample.text))
        report["transformer_subset"] = classification_report(sample.label, transformer, output_dict=True, zero_division=0)
        report["comparison_note"] = "Different training corpora; transformer uses truncated inputs and a subset. Not a controlled benchmark."
        report["classical_same_subset_macro_f1"] = float(f1_score(sample.label, classical_subset, average="macro"))
        report["transformer_subset_macro_f1"] = float(f1_score(sample.label, transformer, average="macro"))
        report["classical_same_subset_accuracy"] = float((classical_subset == sample.label.to_numpy()).mean())
        report["transformer_subset_accuracy"] = float((np.asarray(transformer) == sample.label.to_numpy()).mean())
        report["transformer_subset_size"] = len(sample)
        low, high = bootstrap_f1_difference(sample.label, transformer, classical_subset, seed)
        report["transformer_minus_classical_macro_f1_ci95"] = [low, high]

    predictions.to_csv(out / "predictions.csv", index=True, index_label="test_row")
    with open(out / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Saved outputs to {out}")
    print("Classical macro F1:", round(report["classical"]["macro avg"]["f1-score"], 3))
    if transformer_n:
        print("Macro F1 on same subset — classical:", round(report["classical_same_subset_macro_f1"], 3))
        print("Macro F1 on same subset — transformer:", round(report["transformer_subset_macro_f1"], 3))
        print("95% bootstrap CI of transformer - classical:", [round(v, 3) for v in report["transformer_minus_classical_macro_f1_ci95"]])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400, help="Balanced total number of IMDb training reviews (even integer)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--transformer-n", type=int, default=40, help="0 to skip transformer inference")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs", help="Output directory")
    args = ap.parse_args()
    if args.n < 8 or args.n % 2: ap.error("--n must be even and at least 8")
    if args.transformer_n < 0: ap.error("--transformer-n must be nonnegative")
    main(args.n, args.seed, args.transformer_n, args.out)
