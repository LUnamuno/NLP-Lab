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

## Suggested results discussion (complete after running)

1. What were the most frequent words and bigrams for each class? Any stopword or negation effects?
2. Classical F1 (macro): ___; confusion matrix takeaways: ___.
3. On the *same 40 samples*: classical F1 ___ vs DistilBERT F1 ___; small sample uncertainty ___.
4. Identify three errors and explain possible causes (sarcasm, negation, long reviews, domain shift).
5. Next steps: more examples, cleaner evaluation, a third baseline, optional Gradio demo, or a new-language dataset.

## Source

IMDb dataset available through Hugging Face `stanfordnlp/imdb`; pre-trained transformer `distilbert/distilbert-base-uncased-finetuned-sst-2-english`.
