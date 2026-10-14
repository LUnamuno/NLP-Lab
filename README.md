# NLP Sentiment Analysis Lab

An educational end-to-end NLP exercise on **IMDb movie reviews**: text exploration (token frequency, bigrams, word clouds), a supervised baseline (**TF-IDF + Logistic Regression**), and sentiment inference with an existing **DistilBERT classifier**.

## Run

Python 3.10–3.12 recommended. Internet needed on first run to fetch the public IMDb dataset and pre-trained model.

```bash
python -m venv .venv
source .venv/bin/activate   # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m pytest -q
python -m src.run --n 400 --transformer-n 40
```

Results go to `outputs/` (ignored by Git): frequency tables, bigram tables, word clouds, confusion matrix, and `metrics.json`. Add a few sample plots or a metrics table to this README before publishing, and write your own interpretation and limitations.

## Methodology

- Random seed 42; balanced IMDb sample from the **train** split; stratified 75/25 train/test split.
- Only training data is used for frequency visuals, to avoid leaking exploratory information from test examples.
- Classic baseline: TF-IDF unigrams+bigrams (fit on train only), Logistic Regression; evaluate on held-out test set.
- Transformer: use a previously fine-tuned `distilbert-base-uncased-finetuned-sst-2-english` pipeline, on up to 40 **same held-out** texts with truncation; no fine-tuning is done here.
- Comparison: for fairness within that small subset, compute classical model's F1 on the very same texts. Still **not** a controlled benchmark: DistilBERT was trained on SST-2, IMDb is a different distribution, and reviews may be truncated. Report both sample size and limitation.

## Ethics and caveats

- Word clouds and word-frequency statistics are exploratory; they are not evidence of sentiment classifier performance.
- IMDb movie-review labels are not representative of other applications (customer service, Spanish, Basque, etc.).
- This is a learning project, not a production sentiment API.
- Don't claim to have fine-tuned BERT or deployed a model.

## Results and discussion

Run with `python -m src.run --n 400 --transformer-n 40` (seed 42): 300 training reviews, 100 test reviews, 40 of which were also scored with DistilBERT.

| Model | Macro F1 | Evaluated on |
|---|---|---|
| TF-IDF + Logistic Regression | 0.738 | 100 held-out test reviews |
| TF-IDF + Logistic Regression | 0.680 | the same 40-review subset |
| DistilBERT (SST-2) | 0.792 | the same 40-review subset |

### 1. Most frequent words and bigrams

Frequencies barely separate the classes. In both, the top words are stopwords (`the`, `a`, `and`, `of`, `to`) and `br`, the leftover HTML `<br />` tag. The top bigrams (`br br`, `of the`, `in the`, `this movie`, `this film`) are almost identical in both classes. Polarity words such as `bad`, `great` or `worst` do not appear in the top 30.

Stopwords are deliberately not removed, since negation matters for the classifiers. That choice makes the exploratory tables and word clouds say little about sentiment. To see polarity cues, one would need to filter stopwords and `br`, or compare relative frequencies between classes (e.g. log-odds). With only 300 training reviews the counts are also noisy.

### 2. Classical baseline

- Macro F1 is 0.738 and accuracy 0.74 on 100 test reviews. With a sample this small the uncertainty is roughly ±9 points, so the result is better read as "somewhere around 0.65–0.83".
- The negative class has recall 0.82 and the positive class 0.66, so the model leans toward predicting "negative"; precision goes the other way (0.71 vs 0.79). With 50 reviews per class I would not conclude this is a real bias.
- The model was trained on only 300 reviews. It is a learning baseline, not an upper bound; with the full dataset TF-IDF + Logistic Regression typically performs considerably better.

### 3. Same 40 samples

- Classical F1 is 0.68 vs DistilBERT 0.79. DistilBERT gets about 4 more reviews right out of 40.
- Both models are wrong on 5 reviews, only DistilBERT on 3, and only the classical model on 7. These are differences of a few examples, and with n=40 a single flipped prediction moves F1 by about 2–3 points. DistilBERT looks somewhat better here, but this sample does not demonstrate it.
- The comparison is not controlled: DistilBERT was trained on SST-2 (short sentences), the classical model on 300 IMDb reviews, and long reviews are truncated for DistilBERT.

### 4. Example errors

The causes below are hypotheses based on reading the reviews, not verified.

- **Positive review of *The Running Man*** (classical model wrong). It starts with "the acting and script here isn't even up to par…" but ends praising the idea of the movie. Mixed opinions with negation and concessions are hard for a bag of n-grams.
- **Positive review of a Jafar Panahi film** (DistilBERT wrong). It says "I don't think he's as successful. Not that what he's saying isn't important…". Double negation, with the verdict wrapped in qualifications.
- **Positive review of a Robert Taylor western** (DistilBERT wrong). A plot description with negative vocabulary ("slaughter", "hatred", "tragic") used to appraise the film, not to criticise it. Likely domain shift: the tone of the subject is confused with the tone of the opinion.

In this sample, the reviews where DistilBERT fails are on average shorter (about 220 words) than those it gets right (about 277). With 8 errors I would not draw conclusions from this, and I did not check how many reviews actually exceed the 512-token limit, so truncation cannot be ruled in or out.

### 5. Next steps

- Clean `<br />` and try filtering stopwords for the exploratory analysis only, not for the classifiers.
- Use more data (e.g. `--n 4000`) and cross-validation or bootstrap intervals to estimate uncertainty.
- Evaluate DistilBERT on all 100 test reviews, not just 40.
- Add a third baseline, such as a transformer fine-tuned on IMDb, to separate the effect of domain from the effect of architecture.

## Source

IMDb dataset available through Hugging Face `stanfordnlp/imdb`; pre-trained transformer `distilbert/distilbert-base-uncased-finetuned-sst-2-english`.
