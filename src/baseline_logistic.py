from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATASETS = {
    "structural": (
        ROOT
        / "data"
        / "processed"
        / "pandas_ml_structural.csv"
    ),

    "extended": (
        ROOT
        / "data"
        / "processed"
        / "pandas_ml_extended.csv"
    ),

    "stable": (
        ROOT
        / "data"
        / "processed"
        / "pandas_ml_stable.csv"
    ),
}


OUTPUT = (
    ROOT
    / "data"
    / "processed"
    / "pandas_baseline_results.csv"
)


TARGET = "observed_defect_90d"

NON_FEATURE_COLUMNS = [
    "pr_number",
    "collection_year",
    TARGET,
]


# ============================================================
# METRICS
# ============================================================

def evaluate(
    dataset_name,
    model_name,
    y_true,
    probabilities,
    predictions,
):

    tn, fp, fn, tp = (
        confusion_matrix(
            y_true,
            predictions,
            labels=[0, 1],
        )
        .ravel()
    )

    return {
        "dataset": (
            dataset_name
        ),

        "model": (
            model_name
        ),

        "n": (
            len(y_true)
        ),

        "positives": (
            int(
                y_true.sum()
            )
        ),

        "pr_auc": (
            average_precision_score(
                y_true,
                probabilities,
            )
        ),

        "roc_auc": (
            roc_auc_score(
                y_true,
                probabilities,
            )
        ),

        "precision": (
            precision_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),

        "recall": (
            recall_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),

        "f1": (
            f1_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),

        "balanced_accuracy": (
            balanced_accuracy_score(
                y_true,
                predictions,
            )
        ),

        "mcc": (
            matthews_corrcoef(
                y_true,
                predictions,
            )
        ),

        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


# ============================================================
# RUN
# ============================================================

results = []


print()
print("=" * 72)
print("BASELINE TEMPORAL — LOGISTIC REGRESSION")
print("=" * 72)


for dataset_name, dataset_path in (
    DATASETS.items()
):

    print()
    print("=" * 72)
    print(
        dataset_name.upper()
    )
    print("=" * 72)


    df = pd.read_csv(
        dataset_path
    )


    feature_columns = [
        column
        for column in df.columns
        if column not in (
            NON_FEATURE_COLUMNS
        )
    ]


    # ========================================================
    # TEMPORAL SPLIT
    # ========================================================

    train = df[
        df[
            "collection_year"
        ] == 2022
    ].copy()


    validation = df[
        df[
            "collection_year"
        ] == 2023
    ].copy()


    X_train = train[
        feature_columns
    ]

    y_train = train[
        TARGET
    ].astype(int)


    X_validation = validation[
        feature_columns
    ]

    y_validation = validation[
        TARGET
    ].astype(int)


    print(
        "Features:",
        len(feature_columns),
    )

    print(
        "Treino:",
        len(train),
        "| positivos:",
        int(
            y_train.sum()
        ),
    )

    print(
        "Validação:",
        len(validation),
        "| positivos:",
        int(
            y_validation.sum()
        ),
    )


    # ========================================================
    # DUMMY BASELINE
    # ========================================================

    dummy = DummyClassifier(
        strategy="prior"
    )


    dummy.fit(
        X_train,
        y_train,
    )


    dummy_prob = (
        dummy
        .predict_proba(
            X_validation
        )[:, 1]
    )


    dummy_pred = (
        dummy_prob >= 0.5
    ).astype(int)


    results.append(
        evaluate(
            dataset_name,
            "dummy_prior",
            y_validation,
            dummy_prob,
            dummy_pred,
        )
    )


    # ========================================================
    # LOGISTIC REGRESSION
    # ========================================================

    logistic = Pipeline(
        steps=[
            (
                "scaler",
                RobustScaler(),
            ),

            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=5000,
                    random_state=42,
                ),
            ),
        ]
    )


    logistic.fit(
        X_train,
        y_train,
    )


    logistic_prob = (
        logistic
        .predict_proba(
            X_validation
        )[:, 1]
    )


    logistic_pred = (
        logistic_prob >= 0.5
    ).astype(int)


    results.append(
        evaluate(
            dataset_name,
            "logistic_balanced",
            y_validation,
            logistic_prob,
            logistic_pred,
        )
    )


# ============================================================
# RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)


results_df = (
    results_df
    .sort_values(
        [
            "dataset",
            "model",
        ]
    )
    .reset_index(
        drop=True
    )
)


print()
print("=" * 72)
print("RESULTADOS — VALIDAÇÃO 2023")
print("=" * 72)


display_columns = [
    "dataset",
    "model",
    "pr_auc",
    "roc_auc",
    "precision",
    "recall",
    "f1",
    "balanced_accuracy",
    "mcc",
    "tp",
    "fp",
    "fn",
    "tn",
]


print(
    results_df[
        display_columns
    ]
    .round(4)
    .to_string(
        index=False
    )
)


# ============================================================
# REFERENCE PREVALENCE
# ============================================================

validation_reference = (
    pd.read_csv(
        DATASETS[
            "structural"
        ]
    )
)


validation_reference = (
    validation_reference[
        validation_reference[
            "collection_year"
        ] == 2023
    ]
)


prevalence = (
    validation_reference[
        TARGET
    ].mean()
)


print()
print("=" * 72)
print("REFERÊNCIA")
print("=" * 72)


print(
    "Prevalência positiva em 2023:",
    round(
        prevalence,
        5,
    ),
)


print(
    "PR-AUC de referência aleatória:",
    round(
        prevalence,
        5,
    ),
)


# ============================================================
# SAVE
# ============================================================

results_df.to_csv(
    OUTPUT,
    index=False,
)


print()
print("=" * 72)
print("BASELINE CONCLUÍDO")
print("=" * 72)

print(
    f"Resultados salvos em:\n"
    f"{OUTPUT}"
)