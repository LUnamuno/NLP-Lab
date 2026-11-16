"""Small, reproducible sentiment-analysis experiment on IMDb reviews."""
import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, f1_score
from sklearn.model_selection import train_test_split
from wordcloud import WordCloud
from .text_utils import STOPWORDS, exploratory_tokens, strip_html_breaks, top_ngrams

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


def bootstrap_macro_f1(y_true, y_pred, n_boot: int, seed: int) -> dict:
    """Percentile 95% CI for macro F1 by resampling the evaluation set."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    scores = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        scores[i] = f1_score(y_true[idx], y_pred[idx], average="macro", zero_division=0)
    low, high = np.percentile(scores, [2.5, 97.5])
    return {
        "n_bootstrap": n_boot,
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1_ci95": [float(low), float(high)],
    }


def transformer_eval_size(transformer_n: int, n_test: int) -> int:
    """0 skips inference; -1 uses the full test set; positive values cap at n_test."""
    if transformer_n == 0:
        return 0
    if transformer_n < 0:
        return n_test
    return min(transformer_n, n_test)


def chunk_token_ids(ids: list[int], content_max: int, stride: int | None = None) -> list[list[int]]:
    """Split a token-id sequence into windows of at most content_max tokens."""
    if content_max < 1:
        raise ValueError("content_max must be positive")
    ids = list(ids)
    if len(ids) <= content_max:
        return [ids]
    stride = content_max if stride is None else stride
    if stride < 1:
        raise ValueError("stride must be positive")
    chunks: list[list[int]] = []
    start = 0
    while True:
        end = min(start + content_max, len(ids))
        chunks.append(ids[start:end])
        if end == len(ids):
            return chunks
        start += stride


def _positive_label_index(id2label: dict) -> int:
    for key, label in id2label.items():
        if str(label).upper() == "POSITIVE":
            return int(key)
    raise ValueError("Could not find a POSITIVE label in the model config")


def predict_full_reviews(pipe, texts: list[str], batch_size: int = 8) -> dict:
    """Score each review using all tokens: average DistilBERT logits across 512-token chunks.

    The first chunk is the same prefix a truncated 512-token forward pass would see.
    """
    import torch

    tokenizer = pipe.tokenizer
    model = pipe.model
    model.eval()
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        raise ValueError("Tokenizer needs a pad token")
    content_max = tokenizer.model_max_length - 2
    stride = max(1, content_max // 2)
    pos_index = _positive_label_index(model.config.id2label)

    previous_max = tokenizer.model_max_length
    tokenizer.model_max_length = int(1e9)
    try:
        body_ids = [
            tokenizer.encode(text, add_special_tokens=False, truncation=False)
            for text in texts
        ]
    finally:
        tokenizer.model_max_length = previous_max

    owners: list[int] = []
    is_first: list[bool] = []
    windows: list[list[int]] = []
    n_chunks = []
    for i, ids in enumerate(body_ids):
        chunks = chunk_token_ids(ids, content_max, stride=stride)
        n_chunks.append(len(chunks))
        for j, chunk in enumerate(chunks):
            owners.append(i)
            is_first.append(j == 0)
            windows.append([tokenizer.cls_token_id, *chunk, tokenizer.sep_token_id])

    logits_sum = np.zeros((len(texts), model.config.num_labels), dtype=np.float64)
    prefix_logits = np.zeros((len(texts), model.config.num_labels), dtype=np.float64)
    device = model.device
    with torch.no_grad():
        for start in range(0, len(windows), batch_size):
            batch = windows[start:start + batch_size]
            max_len = max(len(x) for x in batch)
            input_ids = torch.tensor(
                [x + [pad_id] * (max_len - len(x)) for x in batch],
                dtype=torch.long,
                device=device,
            )
            attention_mask = (input_ids != pad_id).long()
            scores = model(input_ids=input_ids, attention_mask=attention_mask).logits.cpu().numpy()
            for row, owner, first in zip(scores, owners[start:start + batch_size], is_first[start:start + batch_size]):
                logits_sum[owner] += row
                if first:
                    prefix_logits[owner] = row

    full_logits = logits_sum / np.asarray(n_chunks)[:, None]
    n_long = int(sum(n > 1 for n in n_chunks))
    return {
        "full": (full_logits.argmax(axis=1) == pos_index).astype(int).tolist(),
        "prefix": (prefix_logits.argmax(axis=1) == pos_index).astype(int).tolist(),
        "n_long": n_long,
        "n_chunks_mean": float(np.mean(n_chunks)),
        "max_len": int(tokenizer.model_max_length),
        "n_chunks": n_chunks,
    }


def main(n: int, seed: int, transformer_n: int, bootstrap_n: int):
    OUT.mkdir(exist_ok=True)
    df = load_reviews(n, seed)
    train, test = train_test_split(df, test_size=0.25, stratify=df.label, random_state=seed)

    # Exploratory visuals only: strip <br /> and drop stopwords here.
    # TF-IDF / transformer still see raw reviews (negation and punctuation matter).
    for label, name in ((0, "negative"), (1, "positive")):
        texts = train.loc[train.label == label, "text"].tolist()
        tokens = [tok for text in texts for tok in exploratory_tokens(text)]
        pd.DataFrame(Counter(tokens).most_common(30), columns=["word", "frequency"]).to_csv(OUT / f"words_{name}.csv", index=False)
        pd.DataFrame(top_ngrams(texts, n=2), columns=["bigram", "frequency"]).to_csv(OUT / f"bigrams_{name}.csv", index=False)
        cleaned = " ".join(strip_html_breaks(text) for text in texts)
        wc = WordCloud(width=900, height=450, background_color="white", stopwords=STOPWORDS).generate(cleaned)
        wc.to_file(str(OUT / f"wordcloud_{name}.png"))

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=20000, strip_accents="unicode")
    X_train = vectorizer.fit_transform(train.text)
    X_test = vectorizer.transform(test.text)
    clf = LogisticRegression(max_iter=2000, random_state=seed)
    clf.fit(X_train, train.label)
    classical = clf.predict(X_test)
    report = {
        "n_total": n,
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "classical": classification_report(test.label, classical, output_dict=True, zero_division=0),
        "classical_bootstrap": bootstrap_macro_f1(test.label, classical, bootstrap_n, seed),
    }
    ConfusionMatrixDisplay.from_predictions(test.label, classical, display_labels=["negative", "positive"])
    plt.tight_layout()
    plt.savefig(OUT / "confusion_matrix_classical.png", dpi=150)
    plt.close()

    eval_n = transformer_eval_size(transformer_n, len(test))
    if eval_n:
        from transformers import pipeline
        # This model is fine-tuned on SST-2, not IMDb: treat comparison as illustrative.
        model_id = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
        model = pipeline("text-classification", model=model_id, device=-1)
        sample = test if eval_n == len(test) else test.sample(n=eval_n, random_state=seed)
        texts = sample.text.tolist()
        y = sample.label.to_numpy()
        predicted = predict_full_reviews(model, texts, batch_size=8)
        transformer = np.asarray(predicted["full"])
        prefix = np.asarray(predicted["prefix"])
        classical_same = clf.predict(vectorizer.transform(sample.text))
        short_mask = np.asarray(predicted["n_chunks"]) == 1
        report["transformer"] = classification_report(y, transformer, output_dict=True, zero_division=0)
        report["transformer_bootstrap"] = bootstrap_macro_f1(y, transformer, bootstrap_n, seed)
        report["transformer_prefix"] = classification_report(y, prefix, output_dict=True, zero_division=0)
        report["transformer_prefix_bootstrap"] = bootstrap_macro_f1(y, prefix, bootstrap_n, seed)
        report["classical_same_eval_macro_f1"] = float(f1_score(y, classical_same, average="macro"))
        report["classical_same_eval_bootstrap"] = bootstrap_macro_f1(y, classical_same, bootstrap_n, seed)
        report["transformer_eval_size"] = int(eval_n)
        report["transformer_n_long"] = predicted["n_long"]
        report["transformer_n_chunks_mean"] = predicted["n_chunks_mean"]
        report["transformer_max_len"] = predicted["max_len"]
        if short_mask.any():
            report["n_short"] = int(short_mask.sum())
            report["classical_short_macro_f1"] = float(f1_score(y[short_mask], classical_same[short_mask], average="macro"))
            report["transformer_short_macro_f1"] = float(f1_score(y[short_mask], transformer[short_mask], average="macro"))
            report["classical_short_bootstrap"] = bootstrap_macro_f1(y[short_mask], classical_same[short_mask], bootstrap_n, seed)
            report["transformer_short_bootstrap"] = bootstrap_macro_f1(y[short_mask], transformer[short_mask], bootstrap_n, seed)
        report["comparison_note"] = (
            "Different training corpora (SST-2 vs IMDb). DistilBERT still cannot attend over "
            "a full long review in one pass; long reviews are split into overlapping 512-token "
            "chunks and logits are averaged so both models use every token."
        )
        ConfusionMatrixDisplay.from_predictions(y, transformer, display_labels=["negative", "positive"])
        plt.tight_layout()
        plt.savefig(OUT / "confusion_matrix_transformer.png", dpi=150)
        plt.close()

    with open(OUT / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Saved outputs to {OUT}")
    ci = report["classical_bootstrap"]["macro_f1_ci95"]
    print(
        "Classical macro F1:",
        round(report["classical"]["macro avg"]["f1-score"], 3),
        f"(95% bootstrap CI [{ci[0]:.3f}, {ci[1]:.3f}] on {len(test)} test reviews)",
    )
    if eval_n:
        t_ci = report["transformer_bootstrap"]["macro_f1_ci95"]
        p_ci = report["transformer_prefix_bootstrap"]["macro_f1_ci95"]
        print(
            "Transformer macro F1 (full text, chunked):",
            round(report["transformer"]["macro avg"]["f1-score"], 3),
            f"(95% bootstrap CI [{t_ci[0]:.3f}, {t_ci[1]:.3f}] on {eval_n} reviews; "
            f"{report['transformer_n_long']} longer than {report['transformer_max_len']} tokens)",
        )
        print(
            "Transformer macro F1 (first 512 tokens only):",
            round(report["transformer_prefix"]["macro avg"]["f1-score"], 3),
            f"(95% bootstrap CI [{p_ci[0]:.3f}, {p_ci[1]:.3f}])",
        )
        if "n_short" in report:
            print(
                "Short reviews only — classical:",
                round(report["classical_short_macro_f1"], 3),
                "transformer:",
                round(report["transformer_short_macro_f1"], 3),
                f"(n={report['n_short']})",
            )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000, help="Balanced total number of IMDb training reviews (even integer)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--transformer-n", type=int, default=-1, help="Reviews for DistilBERT: -1=all test, 0=skip, >0=subset")
    ap.add_argument("--bootstrap-n", type=int, default=1000, help="Bootstrap resamples for macro-F1 95% CIs")
    args = ap.parse_args()
    if args.n < 8 or args.n % 2: ap.error("--n must be even and at least 8")
    if args.transformer_n < -1: ap.error("--transformer-n must be -1, 0, or a positive integer")
    if args.bootstrap_n < 100: ap.error("--bootstrap-n must be at least 100")
    main(args.n, args.seed, args.transformer_n, args.bootstrap_n)
