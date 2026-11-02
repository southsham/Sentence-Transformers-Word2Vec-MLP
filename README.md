# Stock News Sentiment: Word2Vec vs. Sentence Transformers

*Completed as part of the Post Graduate Program in Generative AI for Business Applications, UT Austin (2025).*

Classifies daily stock-market news as positive, neutral or negative. The project compares two text-embedding approaches, each paired with a Random Forest and a neural network, and explores how news sentiment relates to price, volume and volatility.

**Stack:** Python · gensim (Word2Vec) · sentence-transformers (`all-MiniLM-L6-v2`) · scikit-learn · TensorFlow/Keras · pandas · matplotlib/seaborn

## Data

349 news items from 2019, each with that day's open, high, low and close prices, trading volume and a sentiment label. Labels are imbalanced: 170 neutral, 99 negative and 80 positive.

## Approach

1. **Exploratory analysis:** sentiment against price, returns, trading volume, daily volatility (high − low) and time.
2. **Embeddings:** a Word2Vec model trained on the corpus, and pretrained `all-MiniLM-L6-v2` sentence embeddings.
3. **Models:** a Random Forest and a Keras neural network (MLP) on each embedding, using a stratified 80/20 train/test split.

## Results

| Model | Test accuracy | Macro F1 |
|---|---|---|
| Word2Vec + Random Forest | 0.443 | 0.337 |
| Sentence Transformer + Random Forest | **0.471** | 0.330 |
| Neural networks (both embeddings) | 0.486 | 0.218 |

**Honest read:**

- Neither embedding produced a usable classifier on a dataset this small.
- The neural networks reached the highest accuracy only by predicting "neutral" for every article, so they scored zero F1 on the positive and negative classes.
- The Random Forests were the only models that learned anything about the minority classes.

**Market insights from the analysis:**

- Negative-news days averaged lower prices.
- Neutral-news days had the *highest* trading volume and volatility, which points to market indecision or broader coverage.
- Return differences between sentiment labels were small.

## What I'd do next

- **More data:** collect a larger corpus across more companies and a longer period. This is the single biggest lever.
- **Class imbalance:** use class weights or resampling so the models stop defaulting to "neutral".
- **Realistic evaluation:** split train and test chronologically, since real forecasting predicts the future.
- **Domain models:** try finance-specific models such as FinBERT.
- **Features:** aggregate sentiment by day or week and combine it with returns and moving averages.
- **Use:** treat sentiment as one risk signal within a broader strategy, never as a standalone forecast.

## Running it

Open `StockMarketNews_Sentiment_Analysis.ipynb` in Google Colab. The dataset isn't included in this repo.
