from pathlib import Path

import pandas as pd

from sklearn.ensemble import (
    RandomForestClassifier,
)
from sklearn.linear_model import (
    LogisticRegression,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    RobustScaler,
)


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
    / "pandas_ranking_evaluation_2023.csv"
)

RANKED_OUTPUT = (
    ROOT
    / "data"
    / "processed"
    / "pandas_ranked_predictions_2023.csv"
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
print("=" * 72)
print("AVALIAÇÃO DE RANKING — 2023")
print("=" * 72)

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


# ============================================================
# MODELS
# ============================================================

models = {

    "logistic_balanced": Pipeline(
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
    ),

    "random_forest": (
        RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )
    ),
}


# ============================================================
# RANKING LEVELS
# ============================================================

TOP_PERCENTAGES = [
    0.01,
    0.02,
    0.05,
    0.10,
    0.20,
]


results = []

ranked_frames = []


prevalence = (
    y_validation.mean()
)


# ============================================================
# TRAIN + RANK
# ============================================================

for model_name, model in (
    models.items()
):

    print()
    print(
        "Treinando:",
        model_name,
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


    ranked = validation[
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
        "model"
    ] = model_name


    ranked = (
        ranked
        .sort_values(
            "probability",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )


    ranked[
        "rank"
    ] = (
        ranked.index + 1
    )


    ranked_frames.append(
        ranked
    )


    total_positives = int(
        ranked[
            TARGET
        ].sum()
    )


    # ========================================================
    # TOP-K
    # ========================================================

    for percentage in (
        TOP_PERCENTAGES
    ):

        k = max(
            1,
            round(
                len(ranked)
                * percentage
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


        precision_at_k = (
            positives_found
            / k
        )


        recall_at_k = (
            positives_found
            / total_positives
        )


        lift_at_k = (
            precision_at_k
            / prevalence
        )


        results.append(
            {
                "model": (
                    model_name
                ),

                "top_pct": (
                    percentage
                    * 100
                ),

                "k": (
                    k
                ),

                "positives_found": (
                    positives_found
                ),

                "total_positives": (
                    total_positives
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


# ============================================================
# RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)


ranked_df = pd.concat(
    ranked_frames,
    ignore_index=True,
)


print()
print("=" * 72)
print("PRECISION / RECALL / LIFT @ K")
print("=" * 72)


print(
    results_df[
        [
            "model",
            "top_pct",
            "k",
            "positives_found",
            "precision_at_k",
            "recall_at_k",
            "lift_at_k",
        ]
    ]
    .round(4)
    .to_string(
        index=False
    )
)


# ============================================================
# DIRECT COMPARISON
# ============================================================

print()
print("=" * 72)
print("COMPARAÇÃO DIRETA")
print("=" * 72)


comparison = (
    results_df
    .pivot(
        index="top_pct",
        columns="model",
        values=[
            "positives_found",
            "precision_at_k",
            "recall_at_k",
            "lift_at_k",
        ],
    )
)


print(
    comparison
    .round(4)
    .to_string()
)


# ============================================================
# SAVE
# ============================================================

results_df.to_csv(
    OUTPUT_FILE,
    index=False,
)


ranked_df.to_csv(
    RANKED_OUTPUT,
    index=False,
)


print()
print("=" * 72)
print("AVALIAÇÃO DE RANKING CONCLUÍDA")
print("=" * 72)

print(
    f"Métricas:\n{OUTPUT_FILE}"
)

print()

print(
    f"Predições ordenadas:\n"
    f"{RANKED_OUTPUT}"
)