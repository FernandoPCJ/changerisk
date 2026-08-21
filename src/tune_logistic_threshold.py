from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    fbeta_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = (
    ROOT
    / "data"
    / "processed"
    / "pandas_ml_structural.csv"
)

OUTPUT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "pandas_logistic_thresholds_2023.csv"
)

SUMMARY_FILE = (
    ROOT
    / "data"
    / "processed"
    / "pandas_logistic_threshold_summary_2023.csv"
)


TARGET = "observed_defect_90d"

NON_FEATURE_COLUMNS = [
    "pr_number",
    "collection_year",
    TARGET,
]


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(
    DATA_FILE
)


features = [
    column
    for column in df.columns
    if column not in NON_FEATURE_COLUMNS
]


train = (
    df[
        df["collection_year"] == 2022
    ]
    .copy()
)


validation = (
    df[
        df["collection_year"] == 2023
    ]
    .copy()
)


X_train = train[
    features
]

y_train = (
    train[
        TARGET
    ]
    .astype(int)
)


X_validation = validation[
    features
]

y_validation = (
    validation[
        TARGET
    ]
    .astype(int)
)


print()
print("=" * 74)
print("AJUSTE DO THRESHOLD — LOGISTIC REGRESSION")
print("=" * 74)

print(
    "Treino 2022:",
    len(train),
    "| positivos:",
    int(y_train.sum()),
)

print(
    "Validação 2023:",
    len(validation),
    "| positivos:",
    int(y_validation.sum()),
)


# ============================================================
# MODEL
# ============================================================

model = Pipeline(
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


model.fit(
    X_train,
    y_train,
)


probabilities = (
    model
    .predict_proba(
        X_validation
    )[:, 1]
)


# ============================================================
# THRESHOLD GRID
# ============================================================

thresholds = np.linspace(
    0.01,
    0.99,
    197,
)


rows = []


for threshold in thresholds:

    predictions = (
        probabilities >= threshold
    ).astype(int)


    tn, fp, fn, tp = (
        confusion_matrix(
            y_validation,
            predictions,
            labels=[0, 1],
        )
        .ravel()
    )


    flagged = int(
        predictions.sum()
    )


    flagged_pct = (
        flagged
        / len(predictions)
        * 100
    )


    precision = precision_score(
        y_validation,
        predictions,
        zero_division=0,
    )


    recall = recall_score(
        y_validation,
        predictions,
        zero_division=0,
    )


    f2 = fbeta_score(
        y_validation,
        predictions,
        beta=2,
        zero_division=0,
    )


    mcc = matthews_corrcoef(
        y_validation,
        predictions,
    )


    balanced_accuracy = (
        balanced_accuracy_score(
            y_validation,
            predictions,
        )
    )


    rows.append(
        {
            "threshold": threshold,
            "flagged_prs": flagged,
            "flagged_pct": flagged_pct,
            "precision": precision,
            "recall": recall,
            "f2": f2,
            "mcc": mcc,
            "balanced_accuracy": (
                balanced_accuracy
            ),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
        }
    )


results = pd.DataFrame(
    rows
)


# ============================================================
# BEST F2
# ============================================================

best_f2 = (
    results
    .sort_values(
        [
            "f2",
            "mcc",
        ],
        ascending=False,
    )
    .iloc[0]
)


# ============================================================
# BEST MCC
# ============================================================

best_mcc = (
    results
    .sort_values(
        [
            "mcc",
            "f2",
        ],
        ascending=False,
    )
    .iloc[0]
)


# ============================================================
# REVIEW BUDGETS
# ============================================================

budget_rows = []


for budget in [
    1,
    2,
    5,
    10,
    20,
]:

    candidates = results[
        results[
            "flagged_pct"
        ] <= budget
    ]


    if candidates.empty:
        continue


    # Dentro do orçamento, prioriza maior recall.
    best = (
        candidates
        .sort_values(
            [
                "recall",
                "precision",
                "threshold",
            ],
            ascending=[
                False,
                False,
                True,
            ],
        )
        .iloc[0]
    )


    budget_rows.append(
        {
            "criterion": (
                f"review_budget_{budget}pct"
            ),

            **best.to_dict(),
        }
    )


# ============================================================
# SUMMARY
# ============================================================

summary_rows = [
    {
        "criterion": "max_f2",
        **best_f2.to_dict(),
    },

    {
        "criterion": "max_mcc",
        **best_mcc.to_dict(),
    },
]


summary_rows.extend(
    budget_rows
)


summary = pd.DataFrame(
    summary_rows
)


print()
print("=" * 74)
print("PONTOS OPERACIONAIS")
print("=" * 74)


display_columns = [
    "criterion",
    "threshold",
    "flagged_prs",
    "flagged_pct",
    "precision",
    "recall",
    "f2",
    "mcc",
    "balanced_accuracy",
    "tp",
    "fp",
    "fn",
    "tn",
]


print(
    summary[
        display_columns
    ]
    .round(4)
    .to_string(
        index=False
    )
)


# ============================================================
# SCORE DISTRIBUTION
# ============================================================

positive_scores = probabilities[
    y_validation.to_numpy() == 1
]

negative_scores = probabilities[
    y_validation.to_numpy() == 0
]


print()
print("=" * 74)
print("DISTRIBUIÇÃO DOS SCORES")
print("=" * 74)


print(
    "Mediana positivos:",
    round(
        float(
            np.median(
                positive_scores
            )
        ),
        4,
    ),
)

print(
    "Mediana negativos:",
    round(
        float(
            np.median(
                negative_scores
            )
        ),
        4,
    ),
)


print(
    "P90 positivos:",
    round(
        float(
            np.quantile(
                positive_scores,
                0.90,
            )
        ),
        4,
    ),
)

print(
    "P90 negativos:",
    round(
        float(
            np.quantile(
                negative_scores,
                0.90,
            )
        ),
        4,
    ),
)


# ============================================================
# SAVE
# ============================================================

results.to_csv(
    OUTPUT_FILE,
    index=False,
)


summary.to_csv(
    SUMMARY_FILE,
    index=False,
)


print()
print("=" * 74)
print("AJUSTE DE THRESHOLD CONCLUÍDO")
print("=" * 74)

print(
    f"Todos os thresholds:\n"
    f"{OUTPUT_FILE}"
)

print()

print(
    f"Resumo:\n"
    f"{SUMMARY_FILE}"
)