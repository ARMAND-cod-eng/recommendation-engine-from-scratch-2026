# Recommendation Engine From Scratch — 2026

![Dashboard](recsys_dashboard_2026.png)

[![Medium](https://img.shields.io/badge/Medium-black?logo=Medium)](https://medium.com/@notuearmand250/i-built-netflixs-algorithm-from-scratch-and-the-results-surprised-me-dc2f52eba1c2)
[![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-blue?logo=kaggle)](https://www.kaggle.com/code/armandjunior/recommendation-engine-from-scratch-amazon-fine-f)
[![Python](https://img.shields.io/badge/Python-3.11-green?logo=python)](https://python.org)
[![NumPy](https://img.shields.io/badge/NumPy-2.0.2-blue?logo=numpy)](https://numpy.org)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## What This Project Builds

5 recommendation systems built from scratch on the
**Amazon Fine Food Reviews** dataset (568,454 reviews),
evaluated with a production-grade framework used at Netflix,
Spotify, and Amazon.

**100% NumPy 2.x compatible — zero pip installs required.**

---

## Results

| System | HR@10 | NDCG@10 | MRR | Bias Score↓ |
|--------|-------|---------|-----|-------------|
| Popularity Baseline | 0.0020 | 0.0006 | 0.0002 | 130.2 |
| SVD (Matrix Factorization) | 0.0120 | 0.0060 | 0.0042 | 100.8 |
| Hybrid Ensemble | 0.0180 | 0.0095 | 0.0069 | 145.9 |
| **User-Based CF** | **0.0260** 🥇 | **0.0141** 🥇 | **0.0106** 🥇 | 110.3 |

**Best system is 12.8× better than random selection.**
SVD achieves RMSE = 0.778 on the 4⭐ bucket —
beating the Netflix Prize winning threshold (0.857).

---

## 🏗️ Architecture
```
Data pipeline (EDA, k-core filtering, sparse matrix)
Popularity Baseline    (Bayesian Average)
User-Based CF          (Cosine similarity, k=30)
SVD                    (scipy.sparse.linalg.svds, k=50)
Content-Based          (TF-IDF, 8K features + bigrams)
Hybrid Ensemble        (SVD×0.40 + CF×0.25 + CB×0.20 + Pop×0.15)
RMSE / MAE evaluation
HR@K, NDCG, MRR       (Leave-One-Out, 500 users)
A/B Test               (T-test + Mann-Whitney + Cohen's d)
Popularity Bias Audit  (4-tier fairness framework)
Executive Dashboard
```

---

##  Key Findings

**1. User-CF beats SVD on sparse data (0.195% density)**
When matrix density is this low, neighborhood signal
is stronger than weak latent factors. Algorithm
sophistication is not a substitute for data density.

**2. The Hybrid is not always better**
Giving SVD the highest weight (0.40) when SVD is
the weakest personalized system pulls the Hybrid
away from its strongest signal (CF).

**3. Accuracy and fairness can coexist**
User-CF is simultaneously the most accurate system
(HR@10=0.026) and the second most fair (Bias=110.3).

**4. Combining systems amplifies dominant bias**
Hybrid Head exposure (56.3%) exceeds every individual
system — three signals independently favor popular
products and cannot be counterbalanced by one.

**5. p-value without effect size is incomplete**
p=0.022 with Cohen's d=0.18 means statistically real
but practically small. This distinction separates
senior from junior data scientists.

---

## Stack

| Library | Version | Role |
|---------|---------|------|
| NumPy | 2.0.2 | Core numerics |
| SciPy | 1.16.3 | Sparse SVD |
| scikit-learn | 1.6.1 | TF-IDF, cosine similarity |
| pandas | 2.3.3 | Data wrangling |
| matplotlib | latest | All visualizations |

---

## How to Run

**Option 1 — Kaggle (recommended, zero setup):**
[Open notebook on Kaggle](https://www.kaggle.com/code/armandjunior/recommendation-engine-from-scratch-amazon-fine-f)
→ Fork → Run All. No installation needed.

**Option 2 — Local:**
```bash
git clone https://github.com/YOUR_USERNAME/recommendation-engine-from-scratch-2026
cd recommendation-engine-from-scratch-2026
pip install numpy scipy scikit-learn pandas matplotlib
jupyter notebook Day2_RecSys_2026_NUMPY2_COMPATIBLE.ipynb
```

---

## 📁 Repository Structure
```
├── Day2_RecSys_2026_NUMPY2_COMPATIBLE.ipynb  ← main notebook
├── recsys_dashboard_2026.png                  ← executive dashboard
└── README.md
```


## 📄 License

MIT License — free to use, modify, and distribute
with attribution.

---

*Dataset: [Amazon Fine Food Reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)
(SNAP, Kaggle)*
```
