from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

STRUCTURAL_FILE = (
    ROOT
    / "data"
    / "processed"
    / "pandas_ml_structural.csv"
)

EXTENDED_FILE = (
    ROOT
    / "data"
    / "processed"
    / "pandas_ml_extended.csv"
)

STABLE_FILE = (
    ROOT
    / "data"
    / "processed"
    / "pandas_ml_stable.csv"
)


OUTPUT_DIR = (
    ROOT
    / "data"
    / "processed"
)


TARGET = (
    "observed_defect_90d"
)


ID_COLUMNS = [
    "pr_number",
    "collection_year",
]


# ============================================================
# LOAD
# ============================================================

datasets = {
    "structural": pd.read_csv(
        STRUCTURAL_FILE
    ),

    "extended": pd.read_csv(
        EXTENDED_FILE
    ),

    "stable": pd.read_csv(
        STABLE_FILE
    ),
}


print()
print("=" * 72)
print("EDA ORIENTADA AO MACHINE LEARNING")
print("=" * 72)


# ============================================================
# BASIC DATASET SUMMARY
# ============================================================

summary_rows = []


for name, df in datasets.items():

    features = [
        column
        for column in df.columns
        if column not in (
            ID_COLUMNS
            + [TARGET]
        )
    ]


    positives = int(
        (
            df[
                TARGET
            ] == 1
        ).sum()
    )


    negatives = int(
        (
            df[
                TARGET
            ] == 0
        ).sum()
    )


    positive_rate = (
        positives
        / len(df)
        * 100
    )


    imbalance_ratio = (
        negatives
        / positives
        if positives > 0
        else np.nan
    )


    summary_rows.append(
        {
            "dataset": (
                name
            ),

            "rows": (
                len(df)
            ),

            "features": (
                len(features)
            ),

            "positives": (
                positives
            ),

            "negatives": (
                negatives
            ),

            "positive_rate_pct": (
                positive_rate
            ),

            "negative_to_positive_ratio": (
                imbalance_ratio
            ),
        }
    )


dataset_summary = pd.DataFrame(
    summary_rows
)


print()
print("=" * 72)
print("RESUMO DOS DATASETS")
print("=" * 72)

print(
    dataset_summary
    .round(3)
    .to_string(
        index=False
    )
)


dataset_summary.to_csv(
    OUTPUT_DIR
    / "pandas_ml_dataset_summary.csv",
    index=False,
)


# ============================================================
# TARGET DISTRIBUTION BY YEAR
# ============================================================

extended = datasets[
    "extended"
].copy()


target_by_year = (
    extended
    .groupby(
        [
            "collection_year",
            TARGET,
        ]
    )
    .size()
    .unstack(
        fill_value=0
    )
)


for value in [0, 1]:

    if value not in (
        target_by_year.columns
    ):

        target_by_year[
            value
        ] = 0


target_by_year = (
    target_by_year[
        [0, 1]
    ]
    .rename(
        columns={
            0: "negative",
            1: "positive",
        }
    )
)


target_by_year[
    "total"
] = (
    target_by_year[
        "negative"
    ]
    +
    target_by_year[
        "positive"
    ]
)


target_by_year[
    "positive_rate_pct"
] = (
    target_by_year[
        "positive"
    ]
    /
    target_by_year[
        "total"
    ]
    * 100
)


print()
print("=" * 72)
print("TARGET POR ANO")
print("=" * 72)

print(
    target_by_year
    .round(3)
)


target_by_year.to_csv(
    OUTPUT_DIR
    / "pandas_ml_target_by_year.csv"
)


# ============================================================
# FEATURE LIST
# ============================================================

features = [
    column
    for column in extended.columns
    if column not in (
        ID_COLUMNS
        + [TARGET]
    )
]


# ============================================================
# DISTRIBUTION / SKEWNESS
# ============================================================

distribution_rows = []


for feature in features:

    series = pd.to_numeric(
        extended[
            feature
        ],
        errors="coerce",
    )


    distribution_rows.append(
        {
            "feature": (
                feature
            ),

            "mean": (
                series.mean()
            ),

            "std": (
                series.std()
            ),

            "min": (
                series.min()
            ),

            "q25": (
                series.quantile(
                    0.25
                )
            ),

            "median": (
                series.median()
            ),

            "q75": (
                series.quantile(
                    0.75
                )
            ),

            "q95": (
                series.quantile(
                    0.95
                )
            ),

            "q99": (
                series.quantile(
                    0.99
                )
            ),

            "max": (
                series.max()
            ),

            "skewness": (
                series.skew()
            ),

            "zero_rate_pct": (
                (
                    series == 0
                ).mean()
                * 100
            ),
        }
    )


distribution = pd.DataFrame(
    distribution_rows
)


distribution[
    "abs_skewness"
] = (
    distribution[
        "skewness"
    ].abs()
)


distribution = (
    distribution
    .sort_values(
        "abs_skewness",
        ascending=False,
    )
    .reset_index(
        drop=True
    )
)


print()
print("=" * 72)
print("FEATURES MAIS ASSIMÉTRICAS")
print("=" * 72)

print(
    distribution[
        [
            "feature",
            "median",
            "q95",
            "q99",
            "max",
            "skewness",
            "zero_rate_pct",
        ]
    ]
    .head(20)
    .round(3)
    .to_string(
        index=False
    )
)


distribution.to_csv(
    OUTPUT_DIR
    / "pandas_ml_feature_distribution.csv",
    index=False,
)


# ============================================================
# POSITIVE VS NEGATIVE
# ============================================================

comparison_rows = []


negative = extended[
    extended[
        TARGET
    ] == 0
]


positive = extended[
    extended[
        TARGET
    ] == 1
]


for feature in features:

    neg = pd.to_numeric(
        negative[
            feature
        ],
        errors="coerce",
    )

    pos = pd.to_numeric(
        positive[
            feature
        ],
        errors="coerce",
    )


    neg_median = (
        neg.median()
    )

    pos_median = (
        pos.median()
    )


    neg_mean = (
        neg.mean()
    )

    pos_mean = (
        pos.mean()
    )


    comparison_rows.append(
        {
            "feature": (
                feature
            ),

            "negative_mean": (
                neg_mean
            ),

            "positive_mean": (
                pos_mean
            ),

            "mean_difference": (
                pos_mean
                - neg_mean
            ),

            "negative_median": (
                neg_median
            ),

            "positive_median": (
                pos_median
            ),

            "median_difference": (
                pos_median
                - neg_median
            ),
        }
    )


comparison = pd.DataFrame(
    comparison_rows
)


# Escala a diferença das medianas pelo IQR global.
# Isso é apenas uma medida exploratória,
# não uma seleção de feature.

iqr_lookup = {}


for feature in features:

    series = pd.to_numeric(
        extended[
            feature
        ],
        errors="coerce",
    )

    iqr = (
        series.quantile(
            0.75
        )
        -
        series.quantile(
            0.25
        )
    )

    iqr_lookup[
        feature
    ] = (
        iqr
    )


comparison[
    "median_difference_iqr"
] = comparison.apply(
    lambda row: (
        row[
            "median_difference"
        ]
        /
        iqr_lookup[
            row[
                "feature"
            ]
        ]
        if (
            iqr_lookup[
                row[
                    "feature"
                ]
            ]
            != 0
        )
        else 0.0
    ),
    axis=1,
)


comparison[
    "abs_median_difference_iqr"
] = (
    comparison[
        "median_difference_iqr"
    ].abs()
)


comparison = (
    comparison
    .sort_values(
        "abs_median_difference_iqr",
        ascending=False,
    )
    .reset_index(
        drop=True
    )
)


print()
print("=" * 72)
print("MAIORES DIFERENÇAS ENTRE POSITIVOS E NEGATIVOS")
print("=" * 72)

print(
    comparison[
        [
            "feature",
            "negative_median",
            "positive_median",
            "median_difference",
            "median_difference_iqr",
        ]
    ]
    .head(20)
    .round(3)
    .to_string(
        index=False
    )
)


comparison.to_csv(
    OUTPUT_DIR
    / "pandas_ml_positive_negative_comparison.csv",
    index=False,
)


# ============================================================
# YEARLY FEATURE MEDIANS
# ============================================================

yearly_medians = (
    extended
    .groupby(
        "collection_year"
    )[
        features
    ]
    .median()
    .T
)


yearly_medians.to_csv(
    OUTPUT_DIR
    / "pandas_ml_feature_medians_by_year.csv"
)


print()
print("=" * 72)
print("MEDIANAS DE FEATURES POR ANO")
print("=" * 72)


important_features = [
    "changed_files",
    "additions",
    "deletions",
    "pr_duration_hours",
    "production_files_changed",
    "test_files_changed",
    "author_prior_prs",
    "author_experience_days",
    "file_prior_changes_mean",
    "file_prior_authors_mean",
]


important_features = [
    feature
    for feature in important_features
    if feature in yearly_medians.index
]


print(
    yearly_medians
    .loc[
        important_features
    ]
    .round(2)
    .to_string()
)


# ============================================================
# EXTREME VALUES
# ============================================================

print()
print("=" * 72)
print("PRs COM MAIOR TAMANHO DE MUDANÇA")
print("=" * 72)


largest = (
    extended[
        [
            "pr_number",
            "collection_year",
            TARGET,
            "changed_files",
            "additions",
            "deletions",
        ]
    ]
    .assign(
        total_churn=lambda data: (
            data[
                "additions"
            ]
            +
            data[
                "deletions"
            ]
        )
    )
    .sort_values(
        "total_churn",
        ascending=False,
    )
    .head(15)
)


print(
    largest.to_string(
        index=False
    )
)


largest.to_csv(
    OUTPUT_DIR
    / "pandas_ml_largest_changes.csv",
    index=False,
)


# ============================================================
# FINAL VALIDATION
# ============================================================

print()
print("=" * 72)
print("EDA CONCLUÍDA")
print("=" * 72)


print(
    "PRs:",
    len(
        extended
    ),
)


print(
    "Positivos:",
    len(
        positive
    ),
)


print(
    "Negativos:",
    len(
        negative
    ),
)


print(
    "Razão negativos/positivo:",
    round(
        len(
            negative
        )
        /
        len(
            positive
        ),
        2,
    ),
)


print(
    "Features analisadas:",
    len(
        features
    ),
)


print()

print(
    "Arquivos gerados em:"
)

print(
    OUTPUT_DIR
)