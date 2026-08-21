from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
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

MODEL_DIR = (
    ROOT
    / "models"
)

MODEL_FILE = (
    MODEL_DIR
    / "changerisk_logistic_structural.joblib"
)

METADATA_FILE = (
    MODEL_DIR
    / "changerisk_logistic_structural_metadata.json"
)


# ============================================================
# CONFIG
# ============================================================

TARGET = "observed_defect_90d"

NON_FEATURE_COLUMNS = [
    "pr_number",
    "collection_year",
    TARGET,
]

TRAIN_YEARS = [
    2022,
    2023,
]

VALIDATION_YEAR = 2023

OUT_OF_TIME_YEARS = [
    2024,
    2025,
]

OPERATIONAL_THRESHOLD = 0.690


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
        df[
            "collection_year"
        ].isin(
            TRAIN_YEARS
        )
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


print()
print("=" * 72)
print("TREINAMENTO DO ARTEFATO FINAL")
print("=" * 72)

print(
    "Dataset: Structural"
)

print(
    "Modelo: Logistic Regression"
)

print(
    "Anos de treino:",
    TRAIN_YEARS,
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

print(
    "Features:",
    len(features),
)


model.fit(
    X_train,
    y_train,
)


# ============================================================
# DEVELOPMENT SCORE DISTRIBUTION
# ============================================================

development_scores = (
    model
    .predict_proba(
        X_train
    )[:, 1]
)


score_reference = {
    "min": float(
        development_scores.min()
    ),

    "p25": float(
        pd.Series(
            development_scores
        ).quantile(
            0.25
        )
    ),

    "median": float(
        pd.Series(
            development_scores
        ).median()
    ),

    "p75": float(
        pd.Series(
            development_scores
        ).quantile(
            0.75
        )
    ),

    "p90": float(
        pd.Series(
            development_scores
        ).quantile(
            0.90
        )
    ),

    "p95": float(
        pd.Series(
            development_scores
        ).quantile(
            0.95
        )
    ),

    "p99": float(
        pd.Series(
            development_scores
        ).quantile(
            0.99
        )
    ),

    "max": float(
        development_scores.max()
    ),
}


# ============================================================
# METADATA
# ============================================================

metadata = {
    "project": "ChangeRisk",

    "repository": (
        "pandas-dev/pandas"
    ),

    "dataset": (
        "structural"
    ),

    "model": (
        "LogisticRegression"
    ),

    "scaler": (
        "RobustScaler"
    ),

    "class_weight": (
        "balanced"
    ),

    "random_state": 42,

    "feature_count": (
        len(features)
    ),

    "features": (
        features
    ),

    "train_years": (
        TRAIN_YEARS
    ),

    "validation_year": (
        VALIDATION_YEAR
    ),

    "out_of_time_test_years": (
        OUT_OF_TIME_YEARS
    ),

    "training_rows": (
        len(train)
    ),

    "training_positives": int(
        y_train.sum()
    ),

    "training_negatives": int(
        (
            y_train == 0
        ).sum()
    ),

    "operational_threshold": (
        OPERATIONAL_THRESHOLD
    ),

    "threshold_note": (
        "Threshold selected on 2023 validation data. "
        "It did not generalize as a fixed binary threshold "
        "to the 2024-2025 out-of-time stress test. "
        "The model should preferably be interpreted as a "
        "risk-ranking model rather than a calibrated "
        "probability classifier."
    ),

    "score_reference": (
        score_reference
    ),
}


# ============================================================
# SAVE
# ============================================================

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


joblib.dump(
    model,
    MODEL_FILE,
)


with open(
    METADATA_FILE,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        metadata,
        file,
        indent=2,
        ensure_ascii=False,
    )


# ============================================================
# VALIDATE ARTIFACT
# ============================================================

loaded_model = joblib.load(
    MODEL_FILE
)


test_scores = (
    loaded_model
    .predict_proba(
        X_train.head(10)
    )[:, 1]
)


if len(test_scores) != 10:

    raise RuntimeError(
        "Falha na validação do artefato."
    )


if not (
    (
        test_scores >= 0
    ).all()
    and
    (
        test_scores <= 1
    ).all()
):

    raise RuntimeError(
        "Scores inválidos após carregar o modelo."
    )


# ============================================================
# RESULT
# ============================================================

print()
print("=" * 72)
print("ARTEFATO CRIADO")
print("=" * 72)

print(
    f"Modelo:\n{MODEL_FILE}"
)

print()

print(
    f"Metadados:\n{METADATA_FILE}"
)

print()

print(
    "Validação de reload: OK"
)

print()

print(
    "Threshold operacional documentado:",
    OPERATIONAL_THRESHOLD,
)

print()

print(
    "IMPORTANTE:"
)

print(
    "O score não deve ser apresentado ao usuário "
    "como probabilidade calibrada de defeito."
)

print(
    "O uso principal será priorização relativa de risco."
)