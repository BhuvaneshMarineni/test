# NLP Assignment 1 - N-Gram Language Model and Perplexity Analysis
Jupyter Notebook Link : https://colab.research.google.com/drive/1XwCrAL4c4sp5hpOjHF_veWbaMfo6n8Mr#scrollTo=5hrwjU-eBxWD
## Overview

This project is focused on building n-gram language models (unigrams, bigrams, and trigrams) using a dataset of headlines. It involves tokenization of text data, generating n-grams, calculating frequency distributions, and evaluating language models using perplexity metrics. The analysis also includes the application of Laplace smoothing to improve perplexity scores.

## Table of Contents

- [Installation](#installation)
- [Dataset](#dataset)
- [Steps Performed](#steps-performed)
- [Perplexity Analysis](#perplexity-analysis)
- [Results](#results)
- [Conclusion](#conclusion)
- [References](#references)

## Installation

To install the necessary dependencies, run the following commands:
```
pip install nltk wordcloud
pip install matplotlib
```

These commands will install the required libraries needed for tokenization, n-gram generation, and plotting.

Dataset
The dataset used for this project consists of headlines extracted from an Excel file. The project utilizes Google Colab to mount Google Drive, where the dataset is stored.

Dataset Path: /content/drive/MyDrive/NLP/all-data.xlsx
The second column of the Excel file contains the headlines used for this analysis
## Steps Performed

1. **Import Libraries and Mount Google Drive:**  
   First, the required libraries are imported, and Google Drive is mounted to access the dataset. This allows seamless interaction with the file stored in Google Drive during the analysis.
from google.colab import drive
drive.mount('/content/drive')


The dataset is loaded from the specified Google Drive path using pandas.read_excel():
```
from google.colab import drive
drive.mount('/content/drive')
```
2. **Load Dataset**  
   The dataset is loaded into a pandas DataFrame for analysis.
```
import pandas as pd
file_path = '/content/drive/MyDrive/NLP/all-data.xlsx'
data = pd.read_excel(file_path)

```
3. **Tokenization**  
   Headlines are tokenized into words using NLTK’s `word_tokenize()` function, and each headline is converted to lowercase for uniform processing.
```
import nltk
from nltk import word_tokenize

nltk.download('punkt')
headlines = data.iloc[:, 1].astype(str).tolist()
tokens_per_headline = [word_tokenize(headline.lower()) for headline in headlines]

```
4. **N-Gram Generation**  
   Unigrams, bigrams, and trigrams are generated using the n-grams function from NLTK.
```
from nltk import ngrams

def generate_ngrams(tokens, n):
    return list(ngrams(tokens, n))

unigrams_per_headline = [generate_ngrams(tokens, 1) for tokens in tokens_per_headline]
bigrams_per_headline = [generate_ngrams(tokens, 2) for tokens in tokens_per_headline]
trigrams_per_headline = [generate_ngrams(tokens, 3) for tokens in tokens_per_headline]

```
5. **Frequency Analysis**  
   The frequency of each n-gram type is calculated using the Counter class from the `collections` module.
```
from collections import Counter

unigram_freq = Counter([ngram for headline in unigrams_per_headline for ngram in headline])
bigram_freq = Counter([ngram for headline in bigrams_per_headline for ngram in headline])
trigram_freq = Counter([ngram for headline in trigrams_per_headline for ngram in headline])

```
6. **Perplexity Calculation**  
   The perplexity of the unigram, bigram, and trigram models is calculated using a custom function. Laplace smoothing is applied to handle zero probabilities.
```
def laplace_smoothing(ngram_freq, n_minus_1_freq, V):
    smoothed_prob = defaultdict(lambda: 1 / (sum(n_minus_1_freq.values()) + V))
    for ngram, count in ngram_freq.items():
        prefix = ngram[:-1]
        smoothed_prob[ngram] = (count + 1) / (n_minus_1_freq[prefix] + V)
    return smoothed_prob
```
7. **Laplace Smoothing**  
   Laplace smoothing is used to improve perplexity scores by avoiding zero probabilities for unseen n-grams.
```
def calculate_perplexity(probabilities, n, tokens):
    N = sum(len(sentence) for sentence in tokens)
    log_prob_sum = 0
    for sentence in tokens:
        for ngram in ngrams(sentence, n):
            prob = probabilities.get(ngram, 1e-10)
            log_prob_sum += np.log(prob)
    return np.exp(-log_prob_sum / N)
```
## Perplexity Analysis

The project evaluates the n-gram language models based on their perplexity scores. Perplexity is calculated both before and after applying Laplace smoothing to demonstrate the model’s predictive power.

- **Unigram Perplexity:** Perplexity of unigram model before and after smoothing.
- **Bigram Perplexity:** Perplexity of bigram model before and after smoothing.
- **Trigram Perplexity:** Perplexity of trigram model before and after smoothing.

## Results

- **N-Gram Frequency Tables:** The frequency of unigrams, bigrams, and trigrams in the dataset is computed.
- **Perplexity Scores:** Perplexity scores for unigram, bigram, and trigram models are compared before and after applying Laplace smoothing.
- **Impact of Laplace Smoothing:** The application of Laplace smoothing generally lowers the perplexity scores, indicating better predictions by the model.

## Conclusion

This project demonstrates how n-gram models can be used to analyze text data and predict word sequences. By applying Laplace smoothing, the perplexity scores are improved, which shows that smoothing techniques can help in better handling unseen n-grams in the dataset.

## References

- NLTK Documentation
- Pandas Documentation
- Matplotlib Documentation
