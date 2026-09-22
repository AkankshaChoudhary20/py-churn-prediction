# py-churn-prediction

A real, trained scikit-learn classification project that predicts customer
churn from account/usage features. Two models (Logistic Regression and
Random Forest) are actually fit on data, evaluated on a held-out test set
with standard classification metrics and an ROC curve, and made scoreable
from the command line.

**Why this project exists in this portfolio:** several other projects here
are "AI" in the sense of wrapping an LLM API (summarization, chat, code
review, etc.). This one is deliberately different — there is no LLM call
anywhere in this repo. It is a classic supervised-learning pipeline: a
labeled dataset, a `ColumnTransformer` feature pipeline, cross-validated
model selection, and an honest evaluation report with real accuracy/
precision/recall/F1/ROC-AUC numbers and a confusion matrix. It exists to
contrast with the LLM-wrapper apps and show statistical model training and
evaluation done properly.

## Important: the data is synthetic, not real

`data/customers.csv` is **generated**, not collected. No real customer data
was used — this environment had no access to a licensed customer/churn
dataset, and publishing real customer records publicly would be a privacy
problem even if one had been available. Instead, `generate_data.py`
fabricates ~5,000 customer rows with a **fully documented, inspectable**
ground-truth relationship between features and the churn label, so the
whole exercise (can a model recover a real signal from noisy data?) is
honest and verifiable by reading the code.

Everything about the generator is seeded (`np.random.seed(42)` plus a
seeded `numpy.random.Generator`), so re-running `generate_data.py` produces
the exact same `data/customers.csv` every time.

### How the synthetic label is generated

Features are drawn with realistic correlations, not independently:

- `tenure_months` — gamma-distributed, skewed toward shorter tenures, capped at 72 months.
- `contract_type` — `month-to-month` / `one-year` / `two-year`, weighted 55/27/18%.
- `monthly_charges` — normal around $60, with a premium added for longer
  contracts (bundled/higher-tier plans).
- `total_charges` — **≈ `tenure_months * monthly_charges` plus Gaussian noise**
  (stddev 40), so it's realistically, not perfectly, consistent with the
  other two.
- `support_tickets_last_90d` — Poisson(λ=1.1).
- `avg_weekly_usage` — normal around 12 hrs/week, reduced by support ticket
  count (frustrated/disengaged users use the product less).
- `payment_method` — `credit_card` / `bank_transfer` / `electronic_check` / `mailed_check`.
- `has_autopay` — boolean, more likely for `credit_card`/`bank_transfer` customers.

The churn label is drawn from an **explicit logistic function** of several
features (see `churn_probability()` in `generate_data.py`):

```
z =  -1.3
     + 1.55 * is_month_to_month
     - 0.75 * is_two_year
     + 0.70 * support_tickets_last_90d       # more tickets -> more churn
     - 0.050 * tenure_months                 # longer tenure -> less churn
     - 0.060 * avg_weekly_usage              # more usage -> less churn
     - 0.85 * has_autopay                    # autopay -> less churn
     + 0.012 * monthly_charges               # pricier plans -> slightly more churn
     + noise ~ Normal(0, 0.55)               # irreducible randomness

p_churn = sigmoid(z)
churned = 1 if random() < p_churn else 0
```

This makes the "ground truth" fully inspectable: month-to-month contracts
and support tickets increase churn risk; tenure, usage, and autopay reduce
it; noise keeps the problem realistically non-deterministic (a model can do
well, but not perfectly — ROC-AUC well below 1.0).

## Pipeline architecture

- **`generate_data.py`** — synthetic data generator, described above. Writes `data/customers.csv`.
- **`features.py`** — a scikit-learn `ColumnTransformer`:
  - `StandardScaler` on numeric features (`tenure_months`, `monthly_charges`,
    `total_charges`, `support_tickets_last_90d`, `avg_weekly_usage`, `has_autopay`)
  - `OneHotEncoder(handle_unknown="ignore")` on categoricals (`contract_type`, `payment_method`)
- **`train.py`** — stratified 80/20 train/test split; builds two full
  `Pipeline`s (preprocessing + model): `LogisticRegression` and
  `RandomForestClassifier`; runs 5-fold `cross_val_score` (ROC-AUC) on the
  training set for model selection; fits both on the full training set and
  saves them with `joblib` to `models/`, along with the held-out test split
  (`models/test_split.joblib`) and the selected best model's name
  (`models/best_model.txt`).
- **`evaluate.py`** — loads the saved pipelines and the held-out test split
  (never used in training or CV), computes accuracy/precision/recall/F1/
  ROC-AUC and a confusion matrix for both models, writes
  `reports/evaluation.md` (metrics table, confusion matrices, written
  analysis, Random Forest's top-5 feature importances) and
  `reports/roc_curve.png`.
- **`predict.py`** — CLI that loads the best saved model and scores one new
  customer record (via CLI flags or a `--json` file), printing churn
  probability and predicted label.
- **`tests/`** — pytest suite:
  - `test_features.py` — checks the `ColumnTransformer` output shape and
    one-hot encoding correctness on a small hand-built sample.
  - `test_data_generator.py` — checks that the generator's label actually
    correlates with the features it's supposed to (churn rate rises with
    support tickets, falls with tenure/usage/autopay/longer contracts,
    `total_charges` correlates strongly with `tenure * monthly_charges`,
    etc.) — this is what would catch a broken/no-signal generator.

## Reproduce it

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

pytest                  # run the test suite
python generate_data.py # regenerate data/customers.csv (deterministic, seed=42)
python train.py         # train + cross-validate + save both models
python evaluate.py      # evaluate on held-out test set, write reports/
python predict.py --tenure_months 3 --monthly_charges 85.5 --total_charges 256.5 \
  --contract_type month-to-month --support_tickets_last_90d 4 \
  --avg_weekly_usage 3.2 --payment_method electronic_check --has_autopay false
```

`data/customers.csv`, `models/*.joblib`, and `reports/` are **committed to
this repo** so it's immediately inspectable without rerunning anything —
you can open `reports/evaluation.md` or `reports/roc_curve.png` right away,
or run `predict.py` directly against the committed models. All of it is
fully reproducible from scratch with the three commands above (the data
generator is seeded, so `data/customers.csv` regenerates identically).

## Actual results (from `reports/evaluation.md`)

5,000 rows generated, 80/20 stratified split (4,000 train / 1,000 test,
~29% churn rate in both). 5-fold CV ROC-AUC on the training set: Logistic
Regression 0.803 (±0.020), Random Forest 0.784 (±0.016) — Logistic
Regression selected as the best model.

Held-out test set metrics:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.754 | 0.628 | 0.387 | 0.479 | 0.771 |
| Random Forest | 0.751 | 0.642 | 0.332 | 0.438 | 0.758 |

**Winner: Logistic Regression** (ROC-AUC 0.771 vs. 0.758). Plausible reason:
the synthetic label is generated directly from a linear combination of the
features passed through a sigmoid, i.e. the true data-generating process is
close to logistic — so a correctly-specified logistic model can match that
structure almost exactly, while the Random Forest has more variance and can
slightly overfit non-existent interactions at this dataset size.

Random Forest top-5 feature importances (which line up with the features
the generator's formula actually weights most heavily):

| Feature | Importance |
|---|---|
| support_tickets_last_90d | 0.2063 |
| tenure_months | 0.1436 |
| contract_type = month-to-month | 0.1387 |
| total_charges | 0.1241 |
| avg_weekly_usage | 0.1150 |

Both models land well above the 0.5 (random-chance) ROC-AUC baseline,
confirming the synthetic generator has real, learnable signal and that both
models actually learned it. Full confusion matrices and the ROC curve plot
are in `reports/evaluation.md` and `reports/roc_curve.png`.

## Project structure

```
generate_data.py     # synthetic data generator (seeded)
features.py           # ColumnTransformer feature pipeline
train.py               # train/CV/select/save both model pipelines
evaluate.py            # test-set evaluation + report + ROC plot
predict.py             # CLI single-record scoring
data/customers.csv     # generated dataset (committed)
models/*.joblib         # fitted pipelines (committed)
reports/evaluation.md   # metrics, analysis, feature importances (committed)
reports/roc_curve.png   # ROC curve plot (committed)
tests/                  # pytest suite
requirements.txt
```

## Requirements

Python 3.14. See `requirements.txt` (pandas, scikit-learn, matplotlib,
joblib, pytest, numpy).
