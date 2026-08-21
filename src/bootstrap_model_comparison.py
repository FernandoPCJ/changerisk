from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
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

OUTPUT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "pandas_bootstrap_model_comparison.csv"
)

SUMMARY_FILE = (
    ROOT
    / "data"
    / "processed"
    / "pandas_bootstrap_model_summary.csv"
)


TARGET = "observed_defect_90d"

NON_FEATURE_COLUMNS = [
    "pr_number",
    "collection_year",
    TARGET,
]


N_BOOTSTRAPS = 2000

RANDOM_SEED = 42

TOP_PERCENTAGES = [
    0.01,
    0.05,
    0.10,
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


X_validation = validation[
    features
]

y_validation = (
    validation[
        TARGET
    ]
    .astype(int)
    .to_numpy()
)


print()
print("=" * 74)
print("BOOTSTRAP PAREADO — COMPARAÇÃO DE MODELOS")
print("=" * 74)

print(
    "Treino:",
    len(train),
)

print(
    "Positivos treino:",
    int(
        y_train.sum()
    ),
)

print(
    "Validação:",
    len(validation),
)

print(
    "Positivos validação:",
    int(
        y_validation.sum()
    ),
)

print(
    "Bootstrap:",
    N_BOOTSTRAPS,
)


# ============================================================
# MODELS
# ============================================================

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


random_forest = (
    RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
)


print()
print("Treinando Logistic Regression...")

logistic.fit(
    X_train,
    y_train,
)


print("Treinando Random Forest...")

random_forest.fit(
    X_train,
    y_train,
)


# ============================================================
# ORIGINAL PREDICTIONS
# ============================================================

prob_logistic = (
    logistic
    .predict_proba(
        X_validation
    )[:, 1]
)


prob_rf = (
    random_forest
    .predict_proba(
        X_validation
    )[:, 1]
)


original_pr_auc_logistic = (
    average_precision_score(
        y_validation,
        prob_logistic,
    )
)


original_pr_auc_rf = (
    average_precision_score(
        y_validation,
        prob_rf,
    )
)


print()
print(
    "PR-AUC Logistic:",
    round(
        original_pr_auc_logistic,
        5,
    ),
)

print(
    "PR-AUC Random Forest:",
    round(
        original_pr_auc_rf,
        5,
    ),
)


# ============================================================
# STRATIFIED BOOTSTRAP INDEXES
# ============================================================

positive_indexes = np.where(
    y_validation == 1
)[0]

negative_indexes = np.where(
    y_validation == 0
)[0]


rng = np.random.default_rng(
    RANDOM_SEED
)


# ============================================================
# RANKING FUNCTION
# ============================================================

def ranking_metrics(
    y_true,
    probabilities,
    percentage,
):

    order = np.argsort(
        probabilities
    )[::-1]


    y_ranked = (
        y_true[
            order
        ]
    )


    k = max(
        1,
        round(
            len(y_ranked)
            * percentage
        ),
    )


    positives_found = int(
        y_ranked[
            :k
        ].sum()
    )


    total_positives = int(
        y_ranked.sum()
    )


    precision = (
        positives_found
        / k
    )


    recall = (
        positives_found
        / total_positives
        if total_positives > 0
        else 0.0
    )


    prevalence = (
        total_positives
        / len(y_ranked)
    )


    lift = (
        precision
        / prevalence
        if prevalence > 0
        else 0.0
    )


    return (
        positives_found,
        precision,
        recall,
        lift,
    )


# ============================================================
# BOOTSTRAP
# ============================================================

rows = []


for iteration in range(
    1,
    N_BOOTSTRAPS + 1,
):

    sampled_positive = (
        rng.choice(
            positive_indexes,
            size=len(
                positive_indexes
            ),
            replace=True,
        )
    )


    sampled_negative = (
        rng.choice(
            negative_indexes,
            size=len(
                negative_indexes
            ),
            replace=True,
        )
    )


    sampled_indexes = np.concatenate(
        [
            sampled_positive,
            sampled_negative,
        ]
    )


    rng.shuffle(
        sampled_indexes
    )


    y_boot = (
        y_validation[
            sampled_indexes
        ]
    )


    logistic_boot = (
        prob_logistic[
            sampled_indexes
        ]
    )


    rf_boot = (
        prob_rf[
            sampled_indexes
        ]
    )


    pr_auc_logistic = (
        average_precision_score(
            y_boot,
            logistic_boot,
        )
    )


    pr_auc_rf = (
        average_precision_score(
            y_boot,
            rf_boot,
        )
    )


    row = {
        "iteration": (
            iteration
        ),

        "logistic_pr_auc": (
            pr_auc_logistic
        ),

        "rf_pr_auc": (
            pr_auc_rf
        ),

        "rf_minus_logistic_pr_auc": (
            pr_auc_rf
            - pr_auc_logistic
        ),
    }


    for percentage in (
        TOP_PERCENTAGES
    ):

        label = int(
            percentage * 100
        )


        (
            logistic_found,
            logistic_precision,
            logistic_recall,
            logistic_lift,
        ) = ranking_metrics(
            y_boot,
            logistic_boot,
            percentage,
        )


        (
            rf_found,
            rf_precision,
            rf_recall,
            rf_lift,
        ) = ranking_metrics(
            y_boot,
            rf_boot,
            percentage,
        )


        row[
            f"logistic_recall_top_{label}"
        ] = logistic_recall

        row[
            f"rf_recall_top_{label}"
        ] = rf_recall

        row[
            f"logistic_lift_top_{label}"
        ] = logistic_lift

        row[
            f"rf_lift_top_{label}"
        ] = rf_lift

        row[
            f"logistic_found_top_{label}"
        ] = logistic_found

        row[
            f"rf_found_top_{label}"
        ] = rf_found


    rows.append(
        row
    )


    if (
        iteration == 1
        or iteration % 250 == 0
        or iteration == N_BOOTSTRAPS
    ):

        print(
            f"[{iteration}/"
            f"{N_BOOTSTRAPS}]"
        )


bootstrap = pd.DataFrame(
    rows
)


# ============================================================
# SUMMARY HELPERS
# ============================================================

def summarize(
    column,
):

    series = (
        bootstrap[
            column
        ]
    )


    return {
        "metric": (
            column
        ),

        "mean": (
            series.mean()
        ),

        "median": (
            series.median()
        ),

        "ci_2_5": (
            series.quantile(
                0.025
            )
        ),

        "ci_97_5": (
            series.quantile(
                0.975
            )
        ),
    }


summary_columns = [
    "logistic_pr_auc",
    "rf_pr_auc",
    "rf_minus_logistic_pr_auc",
]


for percentage in (
    TOP_PERCENTAGES
):

    label = int(
        percentage * 100
    )

    summary_columns.extend(
        [
            f"logistic_recall_top_{label}",
            f"rf_recall_top_{label}",
            f"logistic_lift_top_{label}",
            f"rf_lift_top_{label}",
        ]
    )


summary = pd.DataFrame(
    [
        summarize(
            column
        )
        for column in (
            summary_columns
        )
    ]
)


# ============================================================
# WIN PROBABILITIES
# ============================================================

rf_pr_auc_win_rate = (
    (
        bootstrap[
            "rf_minus_logistic_pr_auc"
        ] > 0
    ).mean()
)


print()
print("=" * 74)
print("INTERVALOS DE CONFIANÇA — 95%")
print("=" * 74)


print(
    summary
    .round(5)
    .to_string(
        index=False
    )
)


print()
print("=" * 74)
print("COMPARAÇÃO DIRETA")
print("=" * 74)


print(
    "Proporção dos bootstraps em que "
    "Random Forest teve PR-AUC maior:",
    round(
        rf_pr_auc_win_rate,
        4,
    ),
)


difference = (
    bootstrap[
        "rf_minus_logistic_pr_auc"
    ]
)


print(
    "Diferença média PR-AUC "
    "(RF - Logistic):",
    round(
        difference.mean(),
        5,
    ),
)


print(
    "IC 95% da diferença:",
    "[",
    round(
        difference.quantile(
            0.025
        ),
        5,
    ),
    ",",
    round(
        difference.quantile(
            0.975
        ),
        5,
    ),
    "]",
)


# ============================================================
# SAVE
# ============================================================

bootstrap.to_csv(
    OUTPUT_FILE,
    index=False,
)


summary.to_csv(
    SUMMARY_FILE,
    index=False,
)


print()
print("=" * 74)
print("BOOTSTRAP CONCLUÍDO")
print("=" * 74)

print(
    f"Resultados completos:\n"
    f"{OUTPUT_FILE}"
)

print()

print(
    f"Resumo:\n"
    f"{SUMMARY_FILE}"
)