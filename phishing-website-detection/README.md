# Phishing Website Detection

A reproducible machine learning project that flags suspicious websites using URL structure, webpage HTML behavior, and domain registration signals, paired with an interactive Streamlit decision-support prototype.

---

## 1. Problem Statement

Phishing websites mimic trusted brands (banks, cloud portals, e-commerce sites) to steal passwords, payment details, and personal data.

- **Intended Users:** Security Operations Center (SOC) analysts, browser/email security teams, and IT administrators triaging suspicious links.
- **Expected Benefit:** Automatically surface high-risk URLs with a clear probability score and human-readable risk signals so analysts can prioritize investigations faster and warn users before credentials are compromised.

---

## 2. Dataset & Limitations

- **Source:** [UCI Machine Learning Repository — Phishing Websites Dataset (ID 327)](https://archive.ics.uci.edu/ml/datasets/phishing+websites) (Rami, McCluskey, & Thabtah, 2015), cached locally at `data/phishing_websites.csv`.
- **Size:** `11,055` websites (`4,898` phishing [44.31%] and `6,157` legitimate [55.69%]).
- **Features:** `30` integer-encoded features (`-1` = Phishing indicator, `0` = Suspicious, `1` = Legitimate indicator) spanning three groups:
  - **URL / Address Bar (9 features):** IP address in URL, URL length, shortening service, `@` symbol, `//` redirect, prefix/suffix hyphen, sub-domain count, SSL/HTTPS state, non-standard port, `https` token in domain.
  - **Webpage Features (14 features):** External favicon, request URLs, anchor URLs, links in meta/script/link tags, Server Form Handler (`SFH`), submitting to email, abnormal WHOIS URL, redirects, `onMouseOver`, right-click disabled, pop-up windows, hidden iframes.
  - **Domain Information (7 features):** Domain registration length, age of domain, DNS record, web traffic rank, PageRank, Google index, inbound links, statistical blacklist report.
- **Target Variable:** `Result` (raw `-1` = Phishing, `1` = Legitimate; mapped to binary `1` = Phishing, `0` = Legitimate in `src/preprocessing.py`).
- **Missing Values:** `0`.
- **Key Limitations:**
  1. **Historical snapshot (2015):** In 2015, lack of HTTPS (`SSLfinal_State = -1`) was a strong phishing signal. Today, most phishing sites also use free HTTPS certificates (e.g., Let's Encrypt), so a model trained on 2015 data can under-flag modern HTTPS phishing without retraining on current traffic.
  2. **Discretized feature collisions:** Because all 30 features are bucketed into `{-1, 0, 1}`, there are `5,206` duplicate feature vectors in the dataset. Different websites can share identical feature profiles.
  3. **No raw URL text:** The public dataset provides pre-extracted features rather than raw URLs or HTML.

---

## 3. Methodology & Models Compared

- **Data Split:** Stratified `60% / 20% / 20%` split (`6,633` Train / `2,211` Validation / `2,211` Test).
- **Leak-Free Preprocessing (`src/preprocessing.py`):**
  - Removes any ID-like or constant columns.
  - Imputes missing values (`SimpleImputer`), filters zero-variance features (`VarianceThreshold`), and standardizes features (`StandardScaler` for linear/kernel/NB models) fit **only** on the training set.
- **Algorithms Tuned on Validation & Evaluated on Test (`src/train_evaluate.py`):**
  1. **Logistic Regression** (linear baseline)
  2. **Support Vector Machine (SVM)** (RBF & linear kernels)
  3. **Decision Tree** (interpretable rule-based model)
  4. **Naive Bayes** (`GaussianNB`, probabilistic baseline)

---

## 4. Results (Held-Out Test Set, n = 2,211)

Hyperparameters were selected strictly on the **Validation** set (by F1-score, tie-broken by ROC-AUC). **SVM** (`C=10.0, kernel='rbf', gamma='scale'`) was selected as the best model.

| Model | Precision | Recall | F1-Score | ROC-AUC | PR-AUC | Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **SVM (Best)** | **0.9658** | 0.9530 | **0.9594** | **0.9909** | **0.9868** | **0.9643** |
| **Decision Tree** | 0.9408 | 0.9571 | 0.9489 | 0.9673 | 0.9461 | 0.9543 |
| **Logistic Regression** | 0.9318 | 0.9070 | 0.9193 | 0.9775 | 0.9745 | 0.9294 |
| **Naive Bayes** | 0.7494 | **0.9683** | 0.8449 | 0.9658 | 0.9593 | 0.8426 |

Comparison plots are saved in `artifacts/plots/`:
- `artifacts/plots/model_comparison.png`
- `artifacts/plots/roc_pr_curves.png`
- `artifacts/plots/confusion_matrices.png`
- `artifacts/plots/error_analysis_and_importance.png`

---

## 5. Error Analysis

On the `2,211` held-out test websites, the best model (**SVM**) produced:
- **False Positives (Legitimate flagged as Phishing):** `33` (`FPR = 2.68%`)
- **False Negatives (Phishing missed as Legitimate):** `46` (`FNR = 4.70%`)

**What drives errors?**
- **False Positives** occur mainly when legitimate websites lack a trusted SSL certificate (`SSLfinal_State = -1`), use hyphens in domain names (`Prefix_Suffix`), or have multiple subdomains—characteristics common on small personal, internal, or newly launched websites.
- **False Negatives** occur mainly when phishing websites present a valid HTTPS certificate (`SSLfinal_State = 1`), normal-looking anchor ratios (`URL_of_Anchor`), and moderate web traffic—mimicking legitimate infrastructure.
- **Naive Bayes** achieved the highest Recall (`0.9683`) but generated `317` False Positives (`Precision = 0.7494`), showing why accuracy or recall alone is insufficient for security triage.

---

## 6. Ethics & Responsible Use

- **Decision-Support Only:** This prototype is **not** a standalone security gatekeeper. Automated domain blocking without human or multi-signal verification can cause collateral damage (e.g., blocking legitimate small businesses, nonprofits, or newly registered sites).
- **Fairness & Bias:** Features such as `age_of_domain`, `web_traffic`, and `Page_Rank` correlate with organizational size and age. Relying heavily on popularity signals can disproportionately flag legitimate new startups, local businesses, or sites in underserved regions as suspicious.
- **Privacy:** The dataset contains aggregated structural indicators rather than user browsing histories or PII. In any real deployment, URL logging must respect user privacy and avoid storing sensitive query parameters (tokens, emails, session IDs).
- **Adversarial Drift:** Attackers actively adapt to detectors (e.g., adopting valid HTTPS certificates and legitimate hosting). Continuous retraining and layered defenses are required.

---

## 7. How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train, Tune & Evaluate All Models
```bash
python -m src.train_evaluate
```
This caches `data/phishing_websites.csv`, trains and tunes all 4 models on the 60/20/20 split, saves models and metrics in `artifacts/`, and generates plots in `artifacts/plots/`.

### 3. Launch the Streamlit Prototype
```bash
streamlit run app.py
```

### 4. Run Automated Tests
```bash
pytest -v
```

### 5. Explore the Notebook
Open `notebooks/01_exploratory_and_modeling.ipynb` in Jupyter or VS Code.
