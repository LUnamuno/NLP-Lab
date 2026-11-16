# NLP Sentiment Analysis Lab

An educational end-to-end NLP exercise on **IMDb movie reviews**: text exploration (token frequency, bigrams, word clouds), a supervised baseline (**TF-IDF + Logistic Regression**), and sentiment inference with an existing **DistilBERT classifier**.

## Run

Python 3.10–3.12 recommended. Internet needed on first run to fetch the public IMDb dataset and pre-trained model.

```bash
python -m venv .venv
source .venv/bin/activate   # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m pytest -q
python -m src.run --n 4000
```

Results go to `outputs/` (ignored by Git): frequency tables, bigram tables, word clouds, confusion matrices, and `metrics.json`. DistilBERT runs on the full test set by default (`--transformer-n -1`); use `--transformer-n 0` to skip it. Optional: `--bootstrap-n` (default 1000) for macro-F1 95% CIs. Long reviews are split into overlapping 512-token chunks so DistilBERT uses every token, matching TF-IDF.

## Methodology

- Random seed 42; balanced IMDb sample from the **train** split (`--n 4000` by default); stratified 75/25 train/test split.
- Only training data is used for frequency visuals, to avoid leaking exploratory information from test examples. Exploratory tables and word clouds strip IMDb `<br />` tags and English stopwords; classifiers still receive raw reviews.
- Classic baseline: TF-IDF unigrams+bigrams (fit on train only), Logistic Regression; evaluate on held-out test set.
- Uncertainty: percentile **95% bootstrap CIs** for macro F1 by resampling the evaluation predictions (test-set sampling noise around a fixed model, not retraining variance).
- Transformer: use a previously fine-tuned `distilbert-base-uncased-finetuned-sst-2-english` pipeline on the **same held-out test set**; no fine-tuning is done here. DistilBERT's window is 512 tokens. Reviews that fit are scored in one pass. Longer reviews are split into overlapping 512-token chunks; we average the chunk logits so the model uses every token (the first chunk is the same prefix a truncated run would see).
- Comparison: both models are scored on the same reviews and, with chunking, on the same tokens. Still **not** a controlled benchmark: DistilBERT was trained on SST-2, the classical model on IMDb, and chunking is not the same as reading a long review in one pass. Report sample size, how many reviews needed chunking, and that limitation.

## Ethics and caveats

- Word clouds and word-frequency statistics are exploratory; they are not evidence of sentiment classifier performance.
- IMDb movie-review labels are not representative of other applications (customer service, Spanish, Basque, etc.).
- This is a learning project, not a production sentiment API.
- Don't claim to have fine-tuned BERT or deployed a model.

## Results and discussion

Run with `python -m src.run --n 4000` (seed 42, DistilBERT on all test reviews, 1000 bootstrap resamples): 3000 training reviews, 1000 test reviews. 140 of those test reviews (14%) are longer than DistilBERT's 512-token window.

| Model | Macro F1 | 95% bootstrap CI | Text DistilBERT sees | Evaluated on |
|---|---|---|---|---|
| TF-IDF + Logistic Regression | 0.844 | [0.821, 0.865] | full review (bag of n-grams) | 1000 test reviews |
| DistilBERT (SST-2), first 512 tokens | 0.882 | [0.862, 0.901] | prefix only; 140 reviews cut | same 1000 |
| DistilBERT (SST-2), chunked full text | 0.882 | [0.861, 0.901] | every token, in overlapping windows | same 1000 |
| TF-IDF + Logistic Regression | 0.841 | [0.816, 0.864] | full review | 860 reviews that already fit in 512 tokens |
| DistilBERT (SST-2) | 0.891 | [0.869, 0.909] | full review, one window | same 860 |

For comparison, the earlier `--n 400` classical run scored 0.738 on 100 test reviews. The previous DistilBERT score of 0.925 was on a 40-review slice and overstated both the point estimate and how sure we could be.

### 1. Most frequent words and bigrams

After stripping `<br />` and English stopwords **for visuals only**, class differences show up more clearly. Shared domain words still dominate (`movie`, `film`, `like`, `story`, `time`), but polarity cues appear in the top ranks: negative reviews favour `bad`, `don't`, and bigrams such as `waste time`, `bad movie`; positive reviews favour `great`, `love`, `best`, and bigrams such as `good movie`. With 3000 training reviews the ranks are more stable than at n=400, but absolute frequency remains a blunt tool compared with relative measures (e.g. log-odds).

Classifiers still see raw text (including stopwords and negation).

### 2. Classical baseline

- Macro F1 is 0.844 (accuracy 0.844) on 1000 test reviews, with a 95% bootstrap CI of [0.821, 0.865]. That interval is much tighter than the rough ±9-point guess from the n=400 run.
- Class-wise metrics are nearly balanced (negative recall 0.86 / positive 0.83; precision 0.83 / 0.85), so the earlier lean toward "negative" does not persist at this sample size.
- Training on 3000 reviews lifts the classical baseline by about 10 F1 points versus n=400. It is still a learning baseline, not an upper bound on the full IMDb train split.

### 3. DistilBERT, truncation, and a fairer comparison

A **token** is a word or word piece (`watching` may become `watch` + `ing`). DistilBERT, like BERT, can attend over at most **512 tokens** in one forward pass, including the special `[CLS]` / `[SEP]` markers. IMDb reviews are often longer than SST-2 sentences, so a naive pipeline **truncates**: it keeps the beginning and drops the rest. That is unfair to DistilBERT if the verdict is at the end (“…but I loved it”), and unfair to the comparison because TF-IDF already uses the whole review.

140 of 1000 test reviews (14%) exceed 512 tokens. Two ways to put both models on the same text:

1. **Chunking (main setting).** Split long reviews into overlapping 512-token windows, score each window, average the logits. DistilBERT still never sees the whole review *at once*, but both models now use every token.
2. **Short-review subset.** Score only the 860 reviews that already fit. No chunking, no missing tail; both models read the complete text as-is.

Results:

- On all 1000 reviews, DistilBERT macro F1 is 0.882 (accuracy 0.882) whether we use only the first 512 tokens or the chunked full text (CIs [0.862, 0.901] vs [0.861, 0.901]). Feeding the tail barely moves the metric, so truncation does not explain the gap with TF-IDF in this sample.
- That gap is about 4 F1 points (0.882 vs 0.844). The CIs barely overlap. DistilBERT still leans slightly toward “negative” (recall 0.92 vs 0.85 on the positive class).
- On the 860 short reviews, where both models already see the full text, DistilBERT is 0.891 vs classical 0.841 (CIs [0.869, 0.909] vs [0.816, 0.864]). The advantage remains, so the leftover difference is more likely **domain** (SST-2 vs IMDb) than length.

Chunking is a workaround, not a long-context model. A transformer trained on IMDb, or one with a longer window, would be the next way to separate architecture from training data.

### 4. Example errors

The examples below are from the earlier `--n 400` run; they remain useful failure modes, not a fresh error audit of the n=4000 model.

- **Positive review of *The Running Man*** (classical model wrong). It starts with "the acting and script here isn't even up to par…" but ends praising the idea of the movie. Mixed opinions with negation and concessions are hard for a bag of n-grams.
- **Positive review of a Jafar Panahi film** (DistilBERT wrong). It says "I don't think he's as successful. Not that what he's saying isn't important…". Double negation, with the verdict wrapped in qualifications.
- **Positive review of a Robert Taylor western** (DistilBERT wrong). A plot description with negative vocabulary ("slaughter", "hatred", "tragic") used to appraise the film, not to criticise it. Likely domain shift: the tone of the subject is confused with the tone of the opinion.

In that earlier sample, reviews where DistilBERT failed were on average shorter than those it got right; with few errors that pattern is not conclusive. On the n=4000 test set, giving DistilBERT the tokens after position 512 did not change macro F1, which also argues against truncation as the main error source here.

### 5. Next steps

- Compare relative frequencies between classes (e.g. log-odds) instead of raw counts.
- Add a third baseline, such as a transformer fine-tuned on IMDb, to separate the effect of domain from the effect of architecture.

## Source

IMDb dataset available through Hugging Face `stanfordnlp/imdb`; pre-trained transformer `distilbert/distilbert-base-uncased-finetuned-sst-2-english`.
