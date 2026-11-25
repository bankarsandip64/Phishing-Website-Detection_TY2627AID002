# Phishing Website Detection Using Machine Learning

A machine learning-based phishing website detection system that classifies websites as **Phishing** or **Legitimate** using website-related security features. This project compares multiple classification algorithms, evaluates their performance, and provides an interactive Streamlit-based decision-support interface.

---

## Overview

Phishing attacks use fraudulent websites that imitate trusted organizations, banks, cloud platforms, and e-commerce services to steal sensitive information such as passwords, payment details, and personal data. This project applies machine learning techniques to identify suspicious website characteristics and predict potential phishing activity.

**Key Objectives:**
- Detect potentially phishing websites using machine learning.
- Extract meaningful patterns from website security features.
- Compare multiple classification algorithms.
- Evaluate models using standard classification metrics.
- Provide an interactive interface for phishing detection.
- Analyze false positives and false negatives to understand model limitations.

## Features

- Binary classification: Phishing or Legitimate.
- Dataset preprocessing and feature selection.
- Comparison of four machine learning algorithms.
- Hyperparameter selection using a validation dataset.
- Stratified training, validation, and testing.
- Performance evaluation using Precision, Recall, F1-score, Accuracy, ROC-AUC, and PR-AUC.
- Confusion matrices and model comparison visualizations.
- Interactive Streamlit application.
- Automated tests using Pytest.
- Reproducible training and evaluation pipeline.

---

## Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Machine Learning | Scikit-learn |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Web Interface | Streamlit |
| Model Serialization | Joblib |
| Testing | Pytest |
| Development Environment | Jupyter Notebook, VS Code |

---

## Dataset

**Source:** [UCI Machine Learning Repository – Phishing Websites Dataset](https://archive.ics.uci.edu/ml/datasets/phishing+websites)  
**Original Research:** Rami, McCluskey, and Thabtah (2015).

The dataset is stored locally at: `data/phishing_websites.csv`

### Dataset Information

| Property | Value |
|---|---:|
| Total Websites | 11,055 |
| Phishing Websites | 4,898 |
| Legitimate Websites | 6,157 |
| Total Features | 30 |
| Missing Values | 0 |

### Feature Categories

The dataset contains 30 integer-encoded features representing different website characteristics.

**1. URL and Address Bar Features**
- IP address in URL
- URL length
- URL shortening services
- Presence of `@` symbol
- Redirecting using `//`
- Prefix/suffix hyphen
- Number of subdomains
- SSL/HTTPS state
- Non-standard ports
- HTTPS token in domain

**2. Webpage Features**
- External favicon
- Request URLs
- Anchor URLs
- Links in meta, script, and link tags
- Server Form Handler (SFH)
- Form submission to email
- Abnormal WHOIS information
- Website redirects
- Mouse-over behavior
- Disabled right-click
- Pop-up windows
- Hidden iframes

**3. Domain Information Features**
- Domain registration length
- Domain age
- DNS record
- Web traffic rank
- PageRank
- Google index
- Inbound links
- Statistical blacklist information

### Target Variable

The original dataset uses:
- `-1` = Phishing
- `1` = Legitimate

For model training, the target is converted into:
- `1` = Phishing
- `0` = Legitimate

This mapping is implemented in `src/preprocessing.py`.

### Dataset Limitations

1. **Historical Dataset:** The dataset originates from 2015. Modern phishing websites may use valid HTTPS certificates, making some historical security indicators less reliable today.
2. **Discretized Features:** All 30 features are encoded using values `-1`, `0`, and `1`. This can result in identical feature vectors for different websites.
3. **No Raw URL Text:** The original dataset provides pre-extracted numerical features rather than raw website URLs or HTML pages. Therefore, the trained model's reported performance applies to these dataset features and does not establish real-world URL detection performance on arbitrary unseen URLs.

---

## Machine Learning Methodology

The project follows a structured machine learning pipeline.

### 1. Data Preprocessing

Implemented in `src/preprocessing.py`.
- Remove ID-like and constant columns.
- Handle missing values using `SimpleImputer`.
- Remove zero-variance features using `VarianceThreshold`.
- Standardize numerical features using `StandardScaler` where appropriate.
- Fit preprocessing transformations only on training data to prevent data leakage.

### 2. Dataset Splitting

The dataset is divided using stratified sampling:

| Dataset | Percentage | Samples |
|---|---:|---:|
| Training | 60% | 6,633 |
| Validation | 20% | 2,211 |
| Testing | 20% | 2,211 |

Stratification maintains the class distribution across all three subsets.

### 3. Algorithms Compared

Four machine learning approaches were evaluated:

| Algorithm | Purpose |
|---|---|
| Logistic Regression | Linear classification baseline |
| Support Vector Machine (SVM) | Classification using linear and RBF kernels |
| Decision Tree | Interpretable rule-based classification |
| Gaussian Naive Bayes | Probabilistic classification baseline |

### 4. Model Selection

Hyperparameters were selected using the validation dataset. The selection criteria were:
- Primary metric: F1-score
- Tie-breaking metric: ROC-AUC

The best-performing model was selected for final evaluation on the held-out test dataset.

---

## Model Performance

The following results were obtained on the held-out test dataset containing 2,211 samples.

### Model
