from pathlib import Path

import pandas as pd

from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
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
from sklearn.utils.class_weight import compute_sample_weight


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
    / "pandas_model_comparison_2023.csv"
)


TARGET = (
    "observed_defect_90d"
)


NON_FEATURE_COLUMNS = [
    "pr_number",
    "collection_year",
    TARGET,
]


# ============================================================
# EVALUATION
# ============================================================

def evaluate(
    dataset_name,
    model_name,
    y_true,
    probabilities,
):

    predictions = (
        probabilities >= 0.5
    ).astype(int)


    tn, fp, fn, tp = (
        confusion_matrix(
            y_true,
            predictions,
            labels=[0, 1],
        )
        .ravel()
    )


    prevalence = (
        y_true.mean()
    )


    pr_auc = (
        average_precision_score(
            y_true,
            probabilities,
        )
    )


    lift = (
        pr_auc / prevalence
        if prevalence > 0
        else 0
    )


    return {
        "dataset": (
            dataset_name
        ),

        "model": (
            model_name
        ),

        "pr_auc": (
            pr_auc
        ),

        "pr_auc_lift": (
            lift
        ),

        "roc_auc": (
            roc_auc_score(
                y_true,
                probabilities,
            )
        ),

        "precision_05": (
            precision_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),

        "recall_05": (
            recall_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),

        "f1_05": (
            f1_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),

        "balanced_accuracy_05": (
            balanced_accuracy_score(
                y_true,
                predictions,
            )
        ),

        "mcc_05": (
            matthews_corrcoef(
                y_true,
                predictions,
            )
        ),

        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


# ============================================================
# MODELS
# ============================================================

def build_models():

    return {
        "logistic_balanced": (
            Pipeline(
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

        "extra_trees": (
            ExtraTreesClassifier(
                n_estimators=500,
                max_depth=None,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )
        ),

        "gradient_boosting": (
            GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=2,
                random_state=42,
            )
        ),
    }


# ============================================================
# RUN
# ============================================================

results = []


print()
print("=" * 76)
print("COMPARAÇÃO TEMPORAL DE MODELOS")
print("=" * 76)

print(
    "Treino: 2022"
)

print(
    "Validação: 2023"
)

print(
    "2024–2025 permanecem fora da seleção."
)


for dataset_name, path in (
    DATASETS.items()
):

    print()
    print("=" * 76)
    print(
        dataset_name.upper()
    )
    print("=" * 76)


    df = pd.read_csv(
        path
    )


    features = [
        column
        for column in df.columns
        if column not in (
            NON_FEATURE_COLUMNS
        )
    ]


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


    print(
        "Features:",
        len(features),
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


    sample_weights = (
        compute_sample_weight(
            class_weight="balanced",
            y=y_train,
        )
    )


    models = build_models()


    for model_name, model in (
        models.items()
    ):

        print(
            "Treinando:",
            model_name,
        )


        if (
            model_name
            == "gradient_boosting"
        ):

            model.fit(
                X_train,
                y_train,
                sample_weight=sample_weights,
            )

        else:

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


        result = evaluate(
            dataset_name,
            model_name,
            y_validation,
            probabilities,
        )


        results.append(
            result
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
            "pr_auc",
            "roc_auc",
        ],
        ascending=False,
    )
    .reset_index(
        drop=True
    )
)


print()
print("=" * 76)
print("RANKING — VALIDAÇÃO 2023")
print("=" * 76)


display_columns = [
    "dataset",
    "model",
    "pr_auc",
    "pr_auc_lift",
    "roc_auc",
    "precision_05",
    "recall_05",
    "f1_05",
    "mcc_05",
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
# BEST MODEL
# ============================================================

best = (
    results_df.iloc[0]
)


print()
print("=" * 76)
print("MELHOR RESULTADO POR PR-AUC")
print("=" * 76)


print(
    "Dataset:",
    best[
        "dataset"
    ],
)

print(
    "Modelo:",
    best[
        "model"
    ],
)

print(
    "PR-AUC:",
    round(
        best[
            "pr_auc"
        ],
        5,
    ),
)

print(
    "Lift sobre prevalência:",
    round(
        best[
            "pr_auc_lift"
        ],
        2,
    ),
    "x",
)

print(
    "ROC-AUC:",
    round(
        best[
            "roc_auc"
        ],
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
print("=" * 76)
print("COMPARAÇÃO CONCLUÍDA")
print("=" * 76)

print(
    f"Resultados salvos em:\n"
    f"{OUTPUT}"
)