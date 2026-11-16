# NLP Sentiment Analysis Lab

An educational end-to-end NLP exercise on **IMDb movie reviews**: text exploration (token frequency, bigrams, word clouds), a supervised baseline (**TF-IDF + Logistic Regression**), and sentiment inference with an existing **DistilBERT classifier**.

## Run

Python 3.10–3.12 recommended. Internet needed on first run to fetch the public IMDb dataset and pre-trained model.

```bash
python -m venv .venv
source .venv/bin/activate   # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m pytest -q
python -m src.run --n 4000 --transformer-n 40
```

Results go to `outputs/` (ignored by Git): frequency tables, bigram tables, word clouds, confusion matrix, and `metrics.json`. Optional flags: `--bootstrap-n` (default 1000) for macro-F1 95% CIs.

## Methodology

- Random seed 42; balanced IMDb sample from the **train** split (`--n 4000` by default); stratified 75/25 train/test split.
- Only training data is used for frequency visuals, to avoid leaking exploratory information from test examples. Exploratory tables and word clouds strip IMDb `<br />` tags and English stopwords; classifiers still receive raw reviews.
- Classic baseline: TF-IDF unigrams+bigrams (fit on train only), Logistic Regression; evaluate on held-out test set.
- Uncertainty: percentile **95% bootstrap CIs** for macro F1 by resampling the evaluation predictions (test-set sampling noise around a fixed model, not retraining variance).
- Transformer: use a previously fine-tuned `distilbert-base-uncased-finetuned-sst-2-english` pipeline, on up to 40 **same held-out** texts with truncation; no fine-tuning is done here.
- Comparison: for fairness within that small subset, compute classical model's F1 on the very same texts. Still **not** a controlled benchmark: DistilBERT was trained on SST-2, IMDb is a different distribution, and reviews may be truncated. Report both sample size and limitation.

## Ethics and caveats

- Word clouds and word-frequency statistics are exploratory; they are not evidence of sentiment classifier performance.
- IMDb movie-review labels are not representative of other applications (customer service, Spanish, Basque, etc.).
- This is a learning project, not a production sentiment API.
- Don't claim to have fine-tuned BERT or deployed a model.

## Results and discussion

Run with `python -m src.run --n 4000 --transformer-n 40` (seed 42, 1000 bootstrap resamples): 3000 training reviews, 1000 test reviews, 40 of which were also scored with DistilBERT.

| Model | Macro F1 | 95% bootstrap CI | Evaluated on |
|---|---|---|---|
| TF-IDF + Logistic Regression | 0.844 | [0.821, 0.865] | 1000 held-out test reviews |
| TF-IDF + Logistic Regression | 0.798 | [0.670, 0.916] | the same 40-review subset |
| DistilBERT (SST-2) | 0.925 | [0.824, 1.000] | the same 40-review subset |

For comparison, the earlier `--n 400` classical run scored 0.738 on 100 test reviews.

### 1. Most frequent words and bigrams

After stripping `<br />` and English stopwords **for visuals only**, class differences show up more clearly. Shared domain words still dominate (`movie`, `film`, `like`, `story`, `time`), but polarity cues appear in the top ranks: negative reviews favour `bad`, `don't`, and bigrams such as `waste time`, `bad movie`; positive reviews favour `great`, `love`, `best`, and bigrams such as `good movie`. With 3000 training reviews the ranks are more stable than at n=400, but absolute frequency remains a blunt tool compared with relative measures (e.g. log-odds).

Classifiers still see raw text (including stopwords and negation).

### 2. Classical baseline

- Macro F1 is 0.844 (accuracy 0.844) on 1000 test reviews, with a 95% bootstrap CI of [0.821, 0.865]. That interval is much tighter than the rough ±9-point guess from the n=400 run.
- Class-wise metrics are nearly balanced (negative recall 0.86 / positive 0.83; precision 0.83 / 0.85), so the earlier lean toward "negative" does not persist at this sample size.
- Training on 3000 reviews lifts the classical baseline by about 10 F1 points versus n=400. It is still a learning baseline, not an upper bound on the full IMDb train split.

### 3. Same 40 samples

- On the shared subset, classical F1 is 0.80 vs DistilBERT 0.92. The subset CIs are wide ([0.67, 0.92] vs [0.82, 1.00]) because n=40 is tiny: a few flipped labels move F1 by several points, and the intervals overlap.
- DistilBERT looks stronger point-wise here, but this is still not a controlled comparison: SST-2 vs IMDb training, truncation, and a 40-review slice of a 1000-review test set.
- The classical model is now trained on 3000 IMDb reviews, so the subset gap is not explained by the tiny classical sample from the earlier run.

### 4. Example errors

The examples below are from the earlier `--n 400` run; they remain useful failure modes, not a fresh error audit of the n=4000 model.

- **Positive review of *The Running Man*** (classical model wrong). It starts with "the acting and script here isn't even up to par…" but ends praising the idea of the movie. Mixed opinions with negation and concessions are hard for a bag of n-grams.
- **Positive review of a Jafar Panahi film** (DistilBERT wrong). It says "I don't think he's as successful. Not that what he's saying isn't important…". Double negation, with the verdict wrapped in qualifications.
- **Positive review of a Robert Taylor western** (DistilBERT wrong). A plot description with negative vocabulary ("slaughter", "hatred", "tragic") used to appraise the film, not to criticise it. Likely domain shift: the tone of the subject is confused with the tone of the opinion.

In that earlier sample, reviews where DistilBERT failed were on average shorter than those it got right; with few errors that pattern is not conclusive, and truncation at 512 tokens was not measured directly.

### 5. Next steps

- Evaluate DistilBERT on the full held-out test set (1000 reviews at `--n 4000`), not just 40.
- Compare relative frequencies between classes (e.g. log-odds) instead of raw counts.
- Add a third baseline, such as a transformer fine-tuned on IMDb, to separate the effect of domain from the effect of architecture.

## Source

IMDb dataset available through Hugging Face `stanfordnlp/imdb`; pre-trained transformer `distilbert/distilbert-base-uncased-finetuned-sst-2-english`.
