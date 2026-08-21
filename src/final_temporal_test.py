from pathlib import Path

import numpy as np
import pandas as pd

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
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = (
    ROOT
    / "data"
    / "processed"
    / "pandas_ml_structural.csv"
)

RESULTS_FILE = (
    ROOT
    / "data"
    / "processed"
    / "pandas_final_temporal_test.csv"
)

PREDICTIONS_FILE = (
    ROOT
    / "data"
    / "processed"
    / "pandas_final_temporal_predictions.csv"
)

POSITIVE_CASES_FILE = (
    ROOT
    / "data"
    / "processed"
    / "pandas_final_positive_cases.csv"
)


TARGET = "observed_defect_90d"

THRESHOLD = 0.690


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


# ============================================================
# FINAL SPLIT
# ============================================================

train = (
    df[
        df["collection_year"].isin(
            [2022, 2023]
        )
    ]
    .copy()
)


test = (
    df[
        df["collection_year"].isin(
            [2024, 2025]
        )
    ]
    .copy()
    .reset_index(drop=True)
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


X_test = test[
    features
]

y_test = (
    test[
        TARGET
    ]
    .astype(int)
)


print()
print("=" * 76)
print("TESTE TEMPORAL FINAL — OUT-OF-TIME")
print("=" * 76)

print(
    "Dataset: Structural"
)

print(
    "Modelo: Logistic Regression"
)

print(
    "Threshold congelado:",
    THRESHOLD,
)

print()

print(
    "Treino: 2022 + 2023"
)

print(
    "PRs treino:",
    len(train),
)

print(
    "Positivos treino:",
    int(
        y_train.sum()
    ),
)

print()

print(
    "Teste: 2024 + 2025"
)

print(
    "PRs teste:",
    len(test),
)

print(
    "Positivos teste:",
    int(
        y_test.sum()
    ),
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


# ============================================================
# PREDICTIONS
# ============================================================

probabilities = (
    model
    .predict_proba(
        X_test
    )[:, 1]
)


predictions = (
    probabilities >= THRESHOLD
).astype(int)


# ============================================================
# GLOBAL METRICS
# ============================================================

tn, fp, fn, tp = (
    confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1],
    )
    .ravel()
)


prevalence = (
    y_test.mean()
)


pr_auc = (
    average_precision_score(
        y_test,
        probabilities,
    )
)


roc_auc = (
    roc_auc_score(
        y_test,
        probabilities,
    )
)


precision = (
    precision_score(
        y_test,
        predictions,
        zero_division=0,
    )
)


recall = (
    recall_score(
        y_test,
        predictions,
        zero_division=0,
    )
)


f1 = (
    f1_score(
        y_test,
        predictions,
        zero_division=0,
    )
)


balanced_accuracy = (
    balanced_accuracy_score(
        y_test,
        predictions,
    )
)


mcc = (
    matthews_corrcoef(
        y_test,
        predictions,
    )
)


flagged_prs = int(
    predictions.sum()
)


flagged_pct = (
    flagged_prs
    / len(test)
    * 100
)


pr_auc_lift = (
    pr_auc / prevalence
    if prevalence > 0
    else np.nan
)


precision_lift = (
    precision / prevalence
    if prevalence > 0
    else np.nan
)


# ============================================================
# RANKING
# ============================================================

ranked = test[
    [
        "pr_number",
        "collection_year",
        TARGET,
    ]
].copy()


ranked[
    "probability"
] = probabilities


ranked[
    "predicted_positive"
] = predictions


ranked = (
    ranked
    .sort_values(
        [
            "probability",
            "pr_number",
        ],
        ascending=[
            False,
            True,
        ],
    )
    .reset_index(drop=True)
)


ranked[
    "rank"
] = (
    ranked.index + 1
)


ranked[
    "rank_pct"
] = (
    ranked[
        "rank"
    ]
    / len(ranked)
    * 100
)


# ============================================================
# TOP-K
# ============================================================

ranking_rows = []


for top_pct in [
    1,
    2,
    5,
    10,
    20,
]:

    k = max(
        1,
        round(
            len(ranked)
            * (
                top_pct / 100
            )
        ),
    )


    top = (
        ranked
        .head(k)
    )


    positives_found = int(
        top[
            TARGET
        ].sum()
    )


    total_positives = int(
        y_test.sum()
    )


    precision_at_k = (
        positives_found
        / k
    )


    recall_at_k = (
        positives_found
        / total_positives
        if total_positives > 0
        else 0.0
    )


    lift_at_k = (
        precision_at_k
        / prevalence
        if prevalence > 0
        else np.nan
    )


    ranking_rows.append(
        {
            "top_pct": top_pct,
            "k": k,
            "positives_found": (
                positives_found
            ),
            "precision_at_k": (
                precision_at_k
            ),
            "recall_at_k": (
                recall_at_k
            ),
            "lift_at_k": (
                lift_at_k
            ),
        }
    )


ranking = pd.DataFrame(
    ranking_rows
)


# ============================================================
# YEARLY METRICS
# ============================================================

year_rows = []


for year in [
    2024,
    2025,
]:

    mask = (
        test[
            "collection_year"
        ] == year
    )


    y_year = (
        y_test.loc[
            mask
        ]
    )


    probability_year = (
        probabilities[
            mask.to_numpy()
        ]
    )


    prediction_year = (
        predictions[
            mask.to_numpy()
        ]
    )


    tn_y, fp_y, fn_y, tp_y = (
        confusion_matrix(
            y_year,
            prediction_year,
            labels=[0, 1],
        )
        .ravel()
    )


    year_rows.append(
        {
            "year": year,

            "prs": (
                len(y_year)
            ),

            "positives": (
                int(
                    y_year.sum()
                )
            ),

            "pr_auc": (
                average_precision_score(
                    y_year,
                    probability_year,
                )
            ),

            "roc_auc": (
                roc_auc_score(
                    y_year,
                    probability_year,
                )
            ),

            "precision": (
                precision_score(
                    y_year,
                    prediction_year,
                    zero_division=0,
                )
            ),

            "recall": (
                recall_score(
                    y_year,
                    prediction_year,
                    zero_division=0,
                )
            ),

            "tp": int(tp_y),
            "fp": int(fp_y),
            "fn": int(fn_y),
            "tn": int(tn_y),
        }
    )


yearly = pd.DataFrame(
    year_rows
)


# ============================================================
# POSITIVE CASES
# ============================================================

positive_cases = (
    ranked[
        ranked[
            TARGET
        ] == 1
    ]
    .copy()
)


# ============================================================
# PRINT GLOBAL
# ============================================================

print()
print("=" * 76)
print("RESULTADO GLOBAL — 2024 + 2025")
print("=" * 76)

print(
    "Prevalência:",
    round(
        prevalence,
        6,
    ),
)

print(
    "PR-AUC:",
    round(
        pr_auc,
        6,
    ),
)

print(
    "PR-AUC lift:",
    round(
        pr_auc_lift,
        3,
    ),
    "x",
)

print(
    "ROC-AUC:",
    round(
        roc_auc,
        6,
    ),
)

print()

print(
    "Threshold:",
    THRESHOLD,
)

print(
    "PRs sinalizadas:",
    flagged_prs,
)

print(
    "% sinalizadas:",
    round(
        flagged_pct,
        3,
    ),
)

print(
    "Precision:",
    round(
        precision,
        6,
    ),
)

print(
    "Precision lift:",
    round(
        precision_lift,
        3,
    ),
    "x",
)

print(
    "Recall:",
    round(
        recall,
        6,
    ),
)

print(
    "F1:",
    round(
        f1,
        6,
    ),
)

print(
    "Balanced accuracy:",
    round(
        balanced_accuracy,
        6,
    ),
)

print(
    "MCC:",
    round(
        mcc,
        6,
    ),
)

print()

print(
    "TP:",
    int(tp),
)

print(
    "FP:",
    int(fp),
)

print(
    "FN:",
    int(fn),
)

print(
    "TN:",
    int(tn),
)


# ============================================================
# PRINT RANKING
# ============================================================

print()
print("=" * 76)
print("RANKING — TOP K")
print("=" * 76)

print(
    ranking
    .round(4)
    .to_string(
        index=False
    )
)


# ============================================================
# PRINT YEARLY
# ============================================================

print()
print("=" * 76)
print("RESULTADOS POR ANO")
print("=" * 76)

print(
    yearly
    .round(6)
    .to_string(
        index=False
    )
)


# ============================================================
# PRINT POSITIVE CASES
# ============================================================

print()
print("=" * 76)
print("RANK DOS 4 CASOS POSITIVOS")
print("=" * 76)

print(
    positive_cases[
        [
            "pr_number",
            "collection_year",
            "probability",
            "predicted_positive",
            "rank",
            "rank_pct",
        ]
    ]
    .round(4)
    .to_string(
        index=False
    )
)


# ============================================================
# SAVE
# ============================================================

global_result = pd.DataFrame(
    [
        {
            "train_years": (
                "2022-2023"
            ),

            "test_years": (
                "2024-2025"
            ),

            "threshold": (
                THRESHOLD
            ),

            "test_prs": (
                len(test)
            ),

            "test_positives": (
                int(
                    y_test.sum()
                )
            ),

            "prevalence": (
                prevalence
            ),

            "pr_auc": (
                pr_auc
            ),

            "pr_auc_lift": (
                pr_auc_lift
            ),

            "roc_auc": (
                roc_auc
            ),

            "flagged_prs": (
                flagged_prs
            ),

            "flagged_pct": (
                flagged_pct
            ),

            "precision": (
                precision
            ),

            "precision_lift": (
                precision_lift
            ),

            "recall": (
                recall
            ),

            "f1": (
                f1
            ),

            "balanced_accuracy": (
                balanced_accuracy
            ),

            "mcc": (
                mcc
            ),

            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
        }
    ]
)


global_result.to_csv(
    RESULTS_FILE,
    index=False,
)


ranked.to_csv(
    PREDICTIONS_FILE,
    index=False,
)


positive_cases.to_csv(
    POSITIVE_CASES_FILE,
    index=False,
)


print()
print("=" * 76)
print("TESTE TEMPORAL FINAL CONCLUÍDO")
print("=" * 76)

print()
print(
    "IMPORTANTE:"
)

print(
    "2024-2025 contém apenas 4 positivos."
)

print(
    "As métricas deste período devem ser "
    "interpretadas como stress test temporal, "
    "não como estimativa precisa de desempenho."
)

print()

print(
    f"Resultado global:\n"
    f"{RESULTS_FILE}"
)

print()

print(
    f"Predições:\n"
    f"{PREDICTIONS_FILE}"
)

print()

print(
    f"Casos positivos:\n"
    f"{POSITIVE_CASES_FILE}"
)