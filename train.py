"""
train.py
========
Loads data/customers.csv, does a stratified train/test split, and trains
TWO models, each wrapped with the shared feature-preprocessing pipeline
(features.build_preprocessor) inside an sklearn Pipeline:

  1. LogisticRegression  - interpretable baseline
  2. RandomForestClassifier - stronger, non-linear model

For model selection, both pipelines are evaluated with 5-fold
cross_val_score (ROC-AUC) on the training set only, before being refit on
the full training set and saved to models/*.joblib via joblib.

Run:
    python train.py

Outputs:
    models/logistic_regression.joblib
    models/random_forest.joblib
    models/test_split.joblib   (X_test, y_test, saved for evaluate.py)
"""

import os
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline

from features import build_preprocessor, prepare_X_y

RANDOM_STATE = 42
DATA_PATH = os.path.join("data", "customers.csv")
MODELS_DIR = "models"


def load_data():
    df = pd.read_csv(DATA_PATH)
    X, y = prepare_X_y(df)
    return X, y


def build_pipelines():
    logistic_pipeline = Pipeline(steps=[
        ("preprocess", build_preprocessor()),
        ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
    ])

    rf_pipeline = Pipeline(steps=[
        ("preprocess", build_preprocessor()),
        ("model", RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=5,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ])

    return {
        "logistic_regression": logistic_pipeline,
        "random_forest": rf_pipeline,
    }


def main():
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    print(f"Train rows: {len(X_train)}, Test rows: {len(X_test)}")
    print(f"Train churn rate: {y_train.mean():.3f}, Test churn rate: {y_test.mean():.3f}")

    pipelines = build_pipelines()
    os.makedirs(MODELS_DIR, exist_ok=True)

    cv_results = {}
    for name, pipe in pipelines.items():
        scores = cross_val_score(
            pipe, X_train, y_train, cv=5, scoring="roc_auc", n_jobs=-1
        )
        cv_results[name] = scores
        print(f"\n[{name}] 5-fold CV ROC-AUC: mean={scores.mean():.4f} std={scores.std():.4f}")
        print(f"  fold scores: {[round(s, 4) for s in scores]}")

        # Fit on the full training set and persist
        pipe.fit(X_train, y_train)
        out_path = os.path.join(MODELS_DIR, f"{name}.joblib")
        joblib.dump(pipe, out_path)
        print(f"  saved fitted pipeline -> {out_path}")

    # Save the held-out test split for evaluate.py so evaluation always
    # runs on data neither model has seen.
    joblib.dump((X_test, y_test), os.path.join(MODELS_DIR, "test_split.joblib"))
    print(f"\nSaved held-out test split -> {os.path.join(MODELS_DIR, 'test_split.joblib')}")

    best_name = max(cv_results, key=lambda n: cv_results[n].mean())
    print(f"\nBest model by mean CV ROC-AUC: {best_name} "
          f"(mean={cv_results[best_name].mean():.4f})")
    with open(os.path.join(MODELS_DIR, "best_model.txt"), "w") as f:
        f.write(best_name)


if __name__ == "__main__":
    main()
