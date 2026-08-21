from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

V2_FILE = (
    ROOT
    / "data"
    / "processed"
    / "pandas_ml_features_v2.csv"
)

TARGET_FILE = (
    ROOT
    / "data"
    / "processed"
    / "pandas_targets_full.csv"
)

CORRELATION_OUTPUT = (
    ROOT
    / "data"
    / "processed"
    / "pandas_feature_correlations_v2.csv"
)

TEMPORAL_OUTPUT = (
    ROOT
    / "data"
    / "processed"
    / "pandas_feature_temporal_audit_v2.csv"
)

TARGET_CORRELATION_OUTPUT = (
    ROOT
    / "data"
    / "processed"
    / "pandas_feature_target_correlations_v2.csv"
)


# ============================================================
# HELPERS
# ============================================================

def safe_spearman(series_a, series_b):
    """
    Calcula correlação de Spearman utilizando DataFrame.corr().

    Evita o caminho Series.corr(method="spearman"), que apresentou
    incompatibilidade no ambiente atual.

    Retorna NaN quando não existe variabilidade suficiente.
    """

    pair = pd.DataFrame(
        {
            "x": pd.to_numeric(
                series_a,
                errors="coerce",
            ),
            "y": pd.to_numeric(
                series_b,
                errors="coerce",
            ),
        }
    ).dropna()

    if len(pair) < 2:
        return np.nan

    if (
        pair["x"].nunique() <= 1
        or pair["y"].nunique() <= 1
    ):
        return np.nan

    correlation_matrix = pair.corr(
        method="spearman"
    )

    return float(
        correlation_matrix.loc[
            "x",
            "y",
        ]
    )


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(
    V2_FILE
)

targets = pd.read_csv(
    TARGET_FILE
)


print()
print("=" * 72)
print("AUDITORIA DE FEATURES V2")
print("=" * 72)

print(
    "PRs:",
    len(df),
)


# ============================================================
# COLUMN ROLES
# ============================================================

IDENTIFIER_COLUMNS = [
    "pr_number",
]

TEMPORAL_METADATA = [
    "collection_year",
]

TARGET_COLUMN = (
    "observed_defect_90d"
)


feature_columns = [
    column
    for column in df.columns
    if column not in (
        IDENTIFIER_COLUMNS
        + TEMPORAL_METADATA
        + [TARGET_COLUMN]
    )
]


print(
    "Features candidatas:",
    len(feature_columns),
)


# ============================================================
# BASIC CONSISTENCY
# ============================================================

duplicates = int(
    df
    .duplicated(
        subset="pr_number"
    )
    .sum()
)


missing_features = int(
    df[
        feature_columns
    ]
    .isna()
    .sum()
    .sum()
)


finite_matrix = (
    df[
        feature_columns
    ]
    .to_numpy(
        dtype=float
    )
)


non_finite = int(
    (
        ~np.isfinite(
            finite_matrix
        )
    ).sum()
)


print()
print("=" * 72)
print("CONSISTÊNCIA BÁSICA")
print("=" * 72)

print(
    "Duplicatas:",
    duplicates,
)

print(
    "Valores ausentes:",
    missing_features,
)

print(
    "Valores não finitos:",
    non_finite,
)


# ============================================================
# TARGET CONSISTENCY
# ============================================================

reference_target = (
    targets
    .set_index(
        "pr_number"
    )[
        "observed_defect_90d"
    ]
)


mapped_target = (
    df[
        "pr_number"
    ]
    .map(
        reference_target
    )
)


missing_prs_target = int(
    mapped_target
    .isna()
    .sum()
)


valid_target_mapping = (
    mapped_target.notna()
)


target_mismatches = int(
    (
        mapped_target.loc[
            valid_target_mapping
        ]
        .astype("Int64")
        .reset_index(drop=True)
        !=
        df.loc[
            valid_target_mapping,
            TARGET_COLUMN,
        ]
        .astype("Int64")
        .reset_index(drop=True)
    ).sum()
)


print()
print("=" * 72)
print("CONSISTÊNCIA DO TARGET")
print("=" * 72)

print(
    "PRs sem correspondência no target:",
    missing_prs_target,
)

print(
    "Targets divergentes:",
    target_mismatches,
)


# ============================================================
# FORBIDDEN / LEAKAGE COLUMNS
# ============================================================

forbidden_exact = {
    "created_at",
    "updated_at",
    "closed_at",
    "merged_at",
    "observation_end",
    "title",
    "author",
    "labels",
    "comments",
    "review_comments",
    "merge_commit_sha",

    "has_high_confidence_candidate_90d",
    "szz_positive_bugfixes",
    "szz_positive_files",
    "szz_processing_complete",
}


forbidden_patterns = [
    "bugfix",
    "after_merge",
    "future",
    "szz_",
]


forbidden_found = []


for column in feature_columns:

    column_lower = (
        column.lower()
    )

    if column in forbidden_exact:

        forbidden_found.append(
            column
        )

        continue

    if any(
        pattern in column_lower
        for pattern
        in forbidden_patterns
    ):

        forbidden_found.append(
            column
        )


print()
print("=" * 72)
print("AUDITORIA DE DATA LEAKAGE")
print("=" * 72)


if forbidden_found:

    print(
        "Features proibidas encontradas:"
    )

    for column in forbidden_found:

        print(
            " -",
            column,
        )

else:

    print(
        "Nenhuma feature proibida encontrada."
    )


# ============================================================
# HISTORICAL TARGET CONSISTENCY
# ============================================================

author_history_invalid = int(
    (
        df[
            "author_known_prior_defects"
        ]
        >
        df[
            "author_known_prior_labels"
        ]
    ).sum()
)


file_history_invalid = int(
    (
        df[
            "file_known_prior_defects_mean"
        ]
        >
        df[
            "file_known_prior_labels_mean"
        ]
        + 1e-12
    ).sum()
)


print()

print(
    "Defeitos históricos de autor > labels conhecidos:",
    author_history_invalid,
)

print(
    "Defeitos históricos de arquivo > labels conhecidos:",
    file_history_invalid,
)


# ============================================================
# FIRST OUTCOME REVEAL CHECK
# ============================================================

targets[
    "merged_at"
] = pd.to_datetime(
    targets[
        "merged_at"
    ],
    utc=True,
    errors="coerce",
)


if "observation_end" in targets.columns:

    targets[
        "observation_end"
    ] = pd.to_datetime(
        targets[
            "observation_end"
        ],
        utc=True,
        errors="coerce",
    )

else:

    targets[
        "observation_end"
    ] = (
        targets[
            "merged_at"
        ]
        + pd.Timedelta(
            days=90
        )
    )


first_known_outcome = (
    targets[
        "observation_end"
    ]
    .min()
)


early_pr_ids = set(
    targets.loc[
        targets[
            "merged_at"
        ]
        < first_known_outcome,
        "pr_number",
    ]
)


early = df[
    df[
        "pr_number"
    ].isin(
        early_pr_ids
    )
]


known_history_columns = [
    "author_known_prior_labels",
    "author_known_prior_defects",
    "author_known_prior_defect_rate",

    "file_known_prior_labels_mean",
    "file_known_prior_defects_mean",
    "file_known_prior_defects_max",

    "file_known_prior_defect_rate_mean",
    "file_known_prior_defect_rate_max",
]


early_history_nonzero = int(
    (
        early[
            known_history_columns
        ]
        != 0
    )
    .sum()
    .sum()
)


print()

print(
    "Primeiro target historicamente conhecível:",
    first_known_outcome,
)

print(
    "PRs anteriores a esse instante:",
    len(early),
)

print(
    "Valores de target histórico indevidamente "
    "presentes antes desse instante:",
    early_history_nonzero,
)


# ============================================================
# DETERMINISTIC REDUNDANCY
# ============================================================

print()
print("=" * 72)
print("REDUNDÂNCIAS DETERMINÍSTICAS")
print("=" * 72)


churn_relation_errors = int(
    (
        df[
            "code_churn"
        ]
        !=
        (
            df[
                "additions"
            ]
            +
            df[
                "deletions"
            ]
        )
    ).sum()
)


ratio_mask = (
    df[
        "code_churn"
    ] > 0
)


ratio_relation_errors = int(
    (
        ~np.isclose(
            (
                df.loc[
                    ratio_mask,
                    "addition_ratio",
                ]
                +
                df.loc[
                    ratio_mask,
                    "deletion_ratio",
                ]
            ),
            1.0,
            atol=1e-9,
        )
    ).sum()
)


touches_tests_errors = int(
    (
        df[
            "touches_tests"
        ]
        !=
        (
            df[
                "test_files_changed"
            ] > 0
        ).astype(int)
    ).sum()
)


touches_docs_errors = int(
    (
        df[
            "touches_documentation"
        ]
        !=
        (
            df[
                "documentation_files_changed"
            ] > 0
        ).astype(int)
    ).sum()
)


rename_errors = int(
    (
        df[
            "has_file_rename"
        ]
        !=
        (
            df[
                "renamed_files"
            ] > 0
        ).astype(int)
    ).sum()
)


print(
    "code_churn != additions + deletions:",
    churn_relation_errors,
)

print(
    "addition_ratio + deletion_ratio != 1:",
    ratio_relation_errors,
)

print(
    "touches_tests inconsistente:",
    touches_tests_errors,
)

print(
    "touches_documentation inconsistente:",
    touches_docs_errors,
)

print(
    "has_file_rename inconsistente:",
    rename_errors,
)


print()

print(
    "Observação: relações corretas acima indicam "
    "redundância potencial, não erro."
)


# ============================================================
# ZERO VARIANCE
# ============================================================

zero_variance = [
    column
    for column in feature_columns
    if (
        df[
            column
        ].nunique(
            dropna=False
        )
        <= 1
    )
]


print()
print("=" * 72)
print("FEATURES SEM VARIAÇÃO")
print("=" * 72)


if zero_variance:

    for column in zero_variance:

        print(
            column
        )

else:

    print(
        "Nenhuma feature sem variação."
    )


# ============================================================
# INTER-FEATURE CORRELATION
# ============================================================

corr = (
    df[
        feature_columns
    ]
    .corr(
        method="spearman"
    )
)


correlation_rows = []


for i, feature_a in enumerate(
    feature_columns
):

    for feature_b in (
        feature_columns[
            i + 1:
        ]
    ):

        value = corr.loc[
            feature_a,
            feature_b,
        ]


        if pd.isna(
            value
        ):
            continue


        if abs(
            value
        ) >= 0.90:

            correlation_rows.append(
                {
                    "feature_a": (
                        feature_a
                    ),

                    "feature_b": (
                        feature_b
                    ),

                    "spearman": (
                        value
                    ),

                    "abs_spearman": (
                        abs(
                            value
                        )
                    ),
                }
            )


high_corr = pd.DataFrame(
    correlation_rows,
    columns=[
        "feature_a",
        "feature_b",
        "spearman",
        "abs_spearman",
    ],
)


if not high_corr.empty:

    high_corr = (
        high_corr
        .sort_values(
            "abs_spearman",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )


high_corr.to_csv(
    CORRELATION_OUTPUT,
    index=False,
)


print()
print("=" * 72)
print("CORRELAÇÕES ALTAS ENTRE FEATURES")
print("=" * 72)


print(
    "Pares com |Spearman| >= 0.90:",
    len(
        high_corr
    ),
)


if not high_corr.empty:

    print()

    print(
        high_corr[
            [
                "feature_a",
                "feature_b",
                "spearman",
            ]
        ]
        .head(25)
        .to_string(
            index=False
        )
    )


# ============================================================
# TEMPORAL DRIFT
# ============================================================

temporal_rows = []


for feature in feature_columns:

    correlation = safe_spearman(
        df[
            feature
        ],
        df[
            "collection_year"
        ],
    )


    temporal_rows.append(
        {
            "feature": (
                feature
            ),

            "spearman_with_year": (
                correlation
            ),

            "abs_spearman_with_year": (
                abs(
                    correlation
                )
                if not pd.isna(
                    correlation
                )
                else np.nan
            ),
        }
    )


temporal = pd.DataFrame(
    temporal_rows
)


temporal = (
    temporal
    .sort_values(
        "abs_spearman_with_year",
        ascending=False,
        na_position="last",
    )
    .reset_index(
        drop=True
    )
)


temporal[
    "temporal_drift_flag"
] = (
    temporal[
        "abs_spearman_with_year"
    ] >= 0.50
)


temporal.to_csv(
    TEMPORAL_OUTPUT,
    index=False,
)


print()
print("=" * 72)
print("DRIFT / CODIFICAÇÃO TEMPORAL")
print("=" * 72)


print(
    temporal[
        [
            "feature",
            "spearman_with_year",
            "temporal_drift_flag",
        ]
    ]
    .head(20)
    .round(3)
    .to_string(
        index=False
    )
)


print()

print(
    "Features com |Spearman| >= 0.50 com o ano:",
    int(
        temporal[
            "temporal_drift_flag"
        ].sum()
    ),
)


# ============================================================
# FEATURE × TARGET ASSOCIATION
# ============================================================

target_rows = []


for feature in feature_columns:

    correlation = safe_spearman(
        df[
            feature
        ],
        df[
            TARGET_COLUMN
        ],
    )


    target_rows.append(
        {
            "feature": (
                feature
            ),

            "spearman_with_target": (
                correlation
            ),

            "abs_spearman_with_target": (
                abs(
                    correlation
                )
                if not pd.isna(
                    correlation
                )
                else np.nan
            ),
        }
    )


target_corr = pd.DataFrame(
    target_rows
)


target_corr = (
    target_corr
    .sort_values(
        "abs_spearman_with_target",
        ascending=False,
        na_position="last",
    )
    .reset_index(
        drop=True
    )
)


target_corr.to_csv(
    TARGET_CORRELATION_OUTPUT,
    index=False,
)


print()
print("=" * 72)
print("ASSOCIAÇÃO UNIVARIADA COM O TARGET")
print("=" * 72)


print(
    target_corr[
        [
            "feature",
            "spearman_with_target",
        ]
    ]
    .head(15)
    .round(3)
    .to_string(
        index=False
    )
)


# ============================================================
# CRITICAL CHECKS
# ============================================================

critical_checks = {
    "duplicatas": (
        duplicates
    ),

    "features ausentes": (
        missing_features
    ),

    "valores não finitos": (
        non_finite
    ),

    "PRs sem target correspondente": (
        missing_prs_target
    ),

    "targets divergentes": (
        target_mismatches
    ),

    "features proibidas": (
        len(
            forbidden_found
        )
    ),

    "histórico de autor impossível": (
        author_history_invalid
    ),

    "histórico de arquivo impossível": (
        file_history_invalid
    ),

    "target conhecido antes do tempo": (
        early_history_nonzero
    ),
}


failed = {
    name: value
    for name, value in critical_checks.items()
    if value != 0
}


print()
print("=" * 72)
print("RESULTADO FINAL")
print("=" * 72)


if failed:

    print(
        "AUDITORIA REPROVADA"
    )

    print()

    for name, value in (
        failed.items()
    ):

        print(
            f"{name}: {value}"
        )

    raise RuntimeError(
        "Há indícios de inconsistência "
        "ou data leakage no dataset."
    )


print(
    "AUDITORIA DE LEAKAGE APROVADA"
)

print()

print(
    "Atenção: correlações altas e drift temporal "
    "não são tratados como falha automática."
)

print(
    "Eles serão usados na seleção final "
    "de features e no desenho da validação."
)

print()

print(
    f"Correlação entre features:\n"
    f"{CORRELATION_OUTPUT}"
)

print()

print(
    f"Auditoria temporal:\n"
    f"{TEMPORAL_OUTPUT}"
)

print()

print(
    f"Associação com target:\n"
    f"{TARGET_CORRELATION_OUTPUT}"
)