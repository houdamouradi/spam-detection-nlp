# SMS Spam Classification — From Naive Bayes to BERT Embeddings

> Academic project — **INSEA** · *Statistiques Multivariées* · 2025

A comparative study of text classification approaches for SMS spam detection, exploring the progression from classical Bag-of-Words methods to modern transformer-based embeddings, with extended analyses on data drift detection and unsupervised modeling.

---

## Project overview

This project explores **five complementary perspectives** on the same SMS spam/ham classification task, building up from simple statistical models to state-of-the-art language representations:

1. **Bag-of-Words & Naive Bayes** — classical statistical baseline
2. **TF-IDF + multiple classifiers** — LDA, QDA, Logistic Regression, SVM
3. **Word2Vec embeddings** — distributed word representations
4. **BERT embeddings** — contextual transformer-based features
5. **Data drift detection & unsupervised modeling (GMM)** — production-readiness analysis

---

## Stack

`Python` · `scikit-learn` · `numpy` · `gensim` (Word2Vec) · `transformers` (BERT) · `PyTorch` · `matplotlib`

---

## Key results

### Part I — Classical approaches (CountVectorizer / TF-IDF)

| Model | Accuracy |
|---|---:|
| LDA | 84.96 % |
| QDA | 50.39 % |
| Logistic Regression | 96.02 % |
| SVM | 98.06 % |
| **MultinomialNB** | **98.10 %** |
| Optimized SVM (TF-IDF + bigrams) | **98.67 %** |

**Custom Multinomial Naive Bayes** was implemented from scratch with NumPy and matches scikit-learn exactly (96.51 % precision, 89.01 % recall).

### Part II — Word2Vec embeddings

| Representation | Accuracy (Logistic Regression) |
|---|---:|
| Word2Vec (Google News 300d) | 95.98 % |
| CountVectorizer | 97.52 % |

Word2Vec is slightly outperformed by simple bag-of-words on this task — the small SMS vocabulary doesn't fully leverage distributed embeddings.

### Part III — BERT embeddings (`bert-base-uncased`)

| Model | CountVectorizer | **BERT embeddings** |
|---|---:|---:|
| LDA | 84.96 % | **99.14 %** |
| Logistic Regression | 96.02 % | **99.32 %** |
| QDA | 50.39 % | **86.61 %** |
| SVM | 98.06 % | **99.00 %** |

**BERT embeddings improve every single model**, with the best score reaching **99.32 %** accuracy with Logistic Regression.

### Part IV — Data drift detection

A discriminator was trained to distinguish train vs. test BERT embeddings:

| Classifier | Accuracy |
|---|---:|
| Logistic Regression | 49.34 % |
| Linear SVM | 49.10 % |
| LDA | 49.46 % |
| QDA | 49.04 % |
| Random Forest | 48.86 % |

All scores **≈ 50 %** → the classifier performs at chance level, meaning train and test distributions are statistically **indistinguishable** in BERT space. **No data drift detected.**

### Part V — Unsupervised modeling with Gaussian Mixture Models

GMM clustering on BERT + PCA embeddings, evaluated against ground-truth labels (Accuracy, ARI, NMI):

| k | Covariance | Accuracy | ARI | NMI |
|---:|---|---:|---:|---:|
| 2 | full | 63.33 % | 0.071 | 0.159 |
| 2 | diag | 62.80 % | 0.065 | 0.156 |
| **3** | **diag** | **73.69 %** | **0.345** | **0.448** |
| 3 | full | 69.27 % | 0.306 | 0.430 |

A **likelihood-ratio test** confirms that `k=3` is significantly better than `k=2` across all covariance types — suggesting BERT captures **sub-categories within ham messages** (e.g., formal vs. informal styles).

---

## Key takeaways

- **Multinomial Naive Bayes** remains an excellent baseline for spam detection (98 % accuracy with minimal compute)
- **BERT contextual embeddings** boost every classifier above 99 % accuracy
- **No data drift** between training and test sets in this dataset
- **GMM reveals hidden structure** in ham messages beyond a simple binary separation

---

## Author

**Houda MOURADI**

*Project completed individually in the context of the Statistiques Multivariées course.*
