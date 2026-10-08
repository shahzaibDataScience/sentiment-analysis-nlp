# 💬 Sentiment Analysis with NLP

Classify product reviews as **positive** 😊 or **negative** 😞 using TF-IDF features and Logistic Regression — a classic first NLP project.

## What you'll learn

- How to **clean text**: lowercase, remove punctuation and numbers
- What **TF-IDF** does: turns text into numbers, weighting meaningful words higher
- How to split text data and train a **LogisticRegression** classifier
- How to read a **confusion matrix** (not just accuracy)
- How to inspect **model coefficients** to find the most positive/negative words

## Dataset

`data/reviews.csv` — 5000 product reviews, generated with seed 42 (2500 positive + 2500 negative):

| Column | Meaning |
|---|---|
| Review | the review text (about phone / laptop / headphones / watch) |
| Sentiment | `positive` or `negative` |

## How to run

```bash
pip install -r requirements.txt
jupyter notebook sentiment_analysis.ipynb
```

## Key findings

- **High accuracy** on the held-out test set (template-generated data is cleanly separable)
- Confusion matrix shows very few misclassifications
- Top positive words: amazing, fantastic, excellent… / Top negative words: terrible, awful, horrible…
- Demo predictions on 3 brand-new sentences work correctly

## 🎓 Explain it yourself

1. Why can't we feed raw text directly into a machine learning model?
2. Explain TF-IDF in your own words: what do "TF" and "IDF" each do?
3. What does the confusion matrix tell you that accuracy alone does not?
4. A word has a large *negative* coefficient in the model — what does that mean?
5. This dataset was generated from templates. Why might accuracy be lower on real-world reviews?
