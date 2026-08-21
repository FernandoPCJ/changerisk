from pathlib import Path

import numpy as np
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

COEFFICIENTS_FILE = (
    ROOT
    / "data"
    / "processed"
    / "pandas_logistic_coefficients.csv"
)

SCORE_SUMMARY_FILE = (
    ROOT
    / "data"
    / "processed"
    / "pandas_score_distribution_by_year.csv"
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
        df["collection_year"].isin(
            [2022, 2023]
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


model.fit(
    X_train,
    y_train,
)


# ============================================================
# COEFFICIENTS
# ============================================================

coefficients = (
    model[
        "model"
    ]
    .coef_[0]
)


coef_df = pd.DataFrame(
    {
        "feature": features,
        "coefficient": coefficients,
    }
)


coef_df[
    "abs_coefficient"
] = (
    coef_df[
        "coefficient"
    ]
    .abs()
)


coef_df[
    "direction"
] = np.where(
    coef_df[
        "coefficient"
    ] > 0,
    "higher_risk",
    "lower_risk",
)


coef_df = (
    coef_df
    .sort_values(
        "abs_coefficient",
        ascending=False,
    )
    .reset_index(
        drop=True
    )
)


# ============================================================
# SCORE DISTRIBUTION BY YEAR
# ============================================================

score_rows = []


for year in [
    2022,
    2023,
    2024,
    2025,
]:

    year_df = (
        df[
            df[
                "collection_year"
            ] == year
        ]
        .copy()
    )


    probabilities = (
        model
        .predict_proba(
            year_df[
                features
            ]
        )[:, 1]
    )


    positive_mask = (
        year_df[
            TARGET
        ].to_numpy()
        == 1
    )


    negative_mask = (
        ~positive_mask
    )


    score_rows.append(
        {
            "year": year,

            "prs": (
                len(
                    year_df
                )
            ),

            "positives": int(
                positive_mask.sum()
            ),

            "prevalence": (
                positive_mask.mean()
            ),

            "score_mean_all": (
                probabilities.mean()
            ),

            "score_median_all": (
                np.median(
                    probabilities
                )
            ),

            "score_p90_all": (
                np.quantile(
                    probabilities,
                    0.90,
                )
            ),

            "score_mean_positive": (
                probabilities[
                    positive_mask
                ].mean()
                if positive_mask.any()
                else np.nan
            ),

            "score_median_positive": (
                np.median(
                    probabilities[
                        positive_mask
                    ]
                )
                if positive_mask.any()
                else np.nan
            ),

            "score_mean_negative": (
                probabilities[
                    negative_mask
                ].mean()
                if negative_mask.any()
                else np.nan
            ),

            "score_median_negative": (
                np.median(
                    probabilities[
                        negative_mask
                    ]
                )
                if negative_mask.any()
                else np.nan
            ),
        }
    )


score_summary = pd.DataFrame(
    score_rows
)


# ============================================================
# PRINT
# ============================================================

print()
print("=" * 76)
print("INTERPRETABILIDADE — LOGISTIC REGRESSION")
print("=" * 76)


print()
print("TOP 15 FEATURES POR |COEFICIENTE|")
print("-" * 76)


print(
    coef_df[
        [
            "feature",
            "coefficient",
            "direction",
        ]
    ]
    .head(15)
    .round(5)
    .to_string(
        index=False
    )
)


print()
print("=" * 76)
print("DISTRIBUIÇÃO DOS SCORES POR ANO")
print("=" * 76)


print(
    score_summary
    .round(5)
    .to_string(
        index=False
    )
)


# ============================================================
# SAVE
# ============================================================

coef_df.to_csv(
    COEFFICIENTS_FILE,
    index=False,
)


score_summary.to_csv(
    SCORE_SUMMARY_FILE,
    index=False,
)


print()
print("=" * 76)
print("DIAGNÓSTICO CONCLUÍDO")
print("=" * 76)


print(
    f"Coeficientes:\n"
    f"{COEFFICIENTS_FILE}"
)

print()

print(
    f"Scores por ano:\n"
    f"{SCORE_SUMMARY_FILE}"
)