"""Small, reproducible sentiment-analysis experiment on IMDb reviews."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, f1_score
from sklearn.model_selection import train_test_split
from wordcloud import WordCloud
from .text_utils import tokenize, top_ngrams

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"


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


def main(n: int, seed: int, transformer_n: int):
    OUT.mkdir(exist_ok=True)
    df = load_reviews(n, seed)
    train, test = train_test_split(df, test_size=0.25, stratify=df.label, random_state=seed)

    # Exploratory word-level features; avoid using these to rewrite the text
    # passed to TF-IDF or the transformer, since negation/punctuation matters.
    for label, name in ((0, "negative"), (1, "positive")):
        texts = train.loc[train.label == label, "text"].tolist()
        tokens = [tok for text in texts for tok in tokenize(text)]
        from collections import Counter
        pd.DataFrame(Counter(tokens).most_common(30), columns=["word", "frequency"]).to_csv(OUT / f"words_{name}.csv", index=False)
        pd.DataFrame(top_ngrams(texts, n=2), columns=["bigram", "frequency"]).to_csv(OUT / f"bigrams_{name}.csv", index=False)
        wc = WordCloud(width=900, height=450, background_color="white").generate(" ".join(texts))
        wc.to_file(str(OUT / f"wordcloud_{name}.png"))

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=20000, strip_accents="unicode")
    X_train = vectorizer.fit_transform(train.text)
    X_test = vectorizer.transform(test.text)
    clf = LogisticRegression(max_iter=2000, random_state=seed)
    clf.fit(X_train, train.label)
    classical = clf.predict(X_test)
    report = {"classical": classification_report(test.label, classical, output_dict=True, zero_division=0)}
    ConfusionMatrixDisplay.from_predictions(test.label, classical, display_labels=["negative", "positive"])
    plt.tight_layout()
    plt.savefig(OUT / "confusion_matrix_classical.png", dpi=150)
    plt.close()

    if transformer_n:
        from transformers import pipeline
        # This model is fine-tuned on SST-2, not IMDb: treat comparison as illustrative.
        model = pipeline("text-classification", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english", device=-1)
        sample = test.sample(n=min(transformer_n, len(test)), random_state=seed)
        # pipeline truncates long reviews at model max length.
        predicted = model(sample.text.tolist(), truncation=True, batch_size=8)
        transformer = [1 if item["label"].upper() == "POSITIVE" else 0 for item in predicted]
        report["transformer_subset"] = classification_report(sample.label, transformer, output_dict=True, zero_division=0)
        report["comparison_note"] = "Different training corpora; transformer uses truncated inputs and a subset. Not a controlled benchmark."
        report["classical_same_subset_macro_f1"] = float(f1_score(sample.label, clf.predict(vectorizer.transform(sample.text)), average="macro"))
        report["transformer_subset_macro_f1"] = float(f1_score(sample.label, transformer, average="macro"))
        report["transformer_subset_size"] = len(sample)

    with open(OUT / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Saved outputs to {OUT}")
    print("Classical macro F1:", round(report["classical"]["macro avg"]["f1-score"], 3))
    if transformer_n:
        print("Macro F1 on same subset — classical:", round(report["classical_same_subset_macro_f1"], 3))
        print("Macro F1 on same subset — transformer:", round(report["transformer_subset_macro_f1"], 3))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400, help="Balanced total number of IMDb training reviews (even integer)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--transformer-n", type=int, default=40, help="0 to skip transformer inference")
    args = ap.parse_args()
    if args.n < 8 or args.n % 2: ap.error("--n must be even and at least 8")
    if args.transformer_n < 0: ap.error("--transformer-n must be nonnegative")
    main(args.n, args.seed, args.transformer_n)
