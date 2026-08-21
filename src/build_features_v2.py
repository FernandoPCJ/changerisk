from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

FULL_PRS_FILE = (
    ROOT
    / "data"
    / "raw"
    / "pandas_pulls_full_enriched.csv"
)

FILES_FILE = (
    ROOT
    / "data"
    / "raw"
    / "pandas_pr_files_full.csv"
)

TARGET_FILE = (
    ROOT
    / "data"
    / "processed"
    / "pandas_targets_full.csv"
)

V1_FILE = (
    ROOT
    / "data"
    / "processed"
    / "pandas_ml_features_v1.csv"
)

OUTPUT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "pandas_ml_features_v2.csv"
)


# ============================================================
# LOAD
# ============================================================

full_prs = pd.read_csv(
    FULL_PRS_FILE
)

files = pd.read_csv(
    FILES_FILE
)

targets = pd.read_csv(
    TARGET_FILE
)

v1 = pd.read_csv(
    V1_FILE
)


print()
print("=" * 70)
print("FEATURE ENGINEERING V2 — HISTÓRICO TEMPORAL")
print("=" * 70)

print(
    "PRs completas disponíveis:",
    len(full_prs),
)

print(
    "PRs elegíveis:",
    len(targets),
)

print(
    "Dataset V1:",
    len(v1),
)

print(
    "Registros de arquivos:",
    len(files),
)


# ============================================================
# DATE NORMALIZATION
# ============================================================

full_prs["merged_at"] = pd.to_datetime(
    full_prs["merged_at"],
    utc=True,
    errors="coerce",
)

targets["merged_at"] = pd.to_datetime(
    targets["merged_at"],
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
        + pd.Timedelta(days=90)
    )


# ============================================================
# NUMERIC NORMALIZATION
# ============================================================

numeric_columns = [
    "commits",
    "changed_files",
    "additions",
    "deletions",
    "code_churn",
]


for column in numeric_columns:

    full_prs[column] = pd.to_numeric(
        full_prs[column],
        errors="coerce",
    ).fillna(0)


# ============================================================
# KNOWN RECONCILIATION
# ============================================================

# PR #45983:
# GitHub summary returned zeros, but git diff --numstat
# confirmed 4 files, 30 additions and 2 deletions.

mask_45983 = (
    full_prs["pr_number"] == 45983
)

full_prs.loc[
    mask_45983,
    "changed_files",
] = 4

full_prs.loc[
    mask_45983,
    "additions",
] = 30

full_prs.loc[
    mask_45983,
    "deletions",
] = 2

full_prs.loc[
    mask_45983,
    "code_churn",
] = 32


# ============================================================
# BOOLEAN NORMALIZATION
# ============================================================

def normalize_bool(series):

    if pd.api.types.is_bool_dtype(
        series
    ):
        return series

    return (
        series
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("true")
    )


files["is_production_code"] = normalize_bool(
    files["is_production_code"]
)


# ============================================================
# PRODUCTION FILES BY PR
# ============================================================

production_files = files[
    files[
        "is_production_code"
    ]
].copy()


pr_to_production_files = (
    production_files
    .groupby("pr_number")["filename"]
    .apply(
        lambda values: set(
            values.astype(str)
        )
    )
    .to_dict()
)


# ============================================================
# LOOKUPS
# ============================================================

eligible_ids = set(
    targets[
        "pr_number"
    ].astype(int)
)


target_lookup = (
    targets
    .set_index("pr_number")[
        "observed_defect_90d"
    ]
    .astype(int)
    .to_dict()
)


full_pr_lookup = (
    full_prs
    .set_index("pr_number")
)


# ============================================================
# OUTCOME REVEAL QUEUE
# ============================================================

# Um target anterior só pode ser usado como histórico
# quando sua janela de 90 dias já tiver terminado.
#
# Exemplo:
#
# PR A merge:       01/01
# observation_end:  01/04
#
# Uma PR integrada em fevereiro NÃO pode conhecer
# observed_defect_90d de A.
#
# Uma PR integrada depois de 01/04 pode.

outcomes = (
    targets[
        [
            "pr_number",
            "observation_end",
            "observed_defect_90d",
        ]
    ]
    .sort_values(
        [
            "observation_end",
            "pr_number",
        ]
    )
    .reset_index(
        drop=True
    )
)


outcome_pointer = 0


# ============================================================
# AUTHOR HISTORY
# ============================================================

author_prior_prs = defaultdict(int)

author_prior_prod_prs = defaultdict(int)

author_prior_code_churn = defaultdict(float)

author_first_merge = {}


# Targets anteriores cujo desfecho já era observável.

author_known_labels = defaultdict(int)

author_known_defects = defaultdict(int)


# ============================================================
# FILE HISTORY
# ============================================================

file_prior_changes = defaultdict(int)

file_prior_authors = defaultdict(set)


# Histórico de targets já observáveis.

file_known_labels = defaultdict(int)

file_known_defects = defaultdict(int)


# ============================================================
# HELPERS
# ============================================================

def author_key(
    author,
    pr_number,
):
    """
    Retorna identificador seguro do autor.

    Autores ausentes não são agrupados entre si,
    evitando criar histórico artificial entre contas
    desconhecidas.
    """

    if pd.isna(author):

        return (
            f"__unknown_pr_{pr_number}"
        )

    author = str(
        author
    ).strip()

    if not author:

        return (
            f"__unknown_pr_{pr_number}"
        )

    return author


def safe_ratio(
    numerator,
    denominator,
):

    if denominator == 0:
        return 0.0

    return (
        numerator
        / denominator
    )


def mean_or_zero(values):

    if not values:
        return 0.0

    return float(
        np.mean(values)
    )


def max_or_zero(values):

    if not values:
        return 0.0

    return float(
        np.max(values)
    )


# ============================================================
# REVEAL HISTORICAL TARGET
# ============================================================

def reveal_outcome(
    pr_number,
    outcome,
):

    if pr_number not in (
        full_pr_lookup.index
    ):
        return


    prior_pr = (
        full_pr_lookup.loc[
            pr_number
        ]
    )


    author = author_key(
        prior_pr["author"],
        pr_number,
    )


    author_known_labels[
        author
    ] += 1


    if outcome == 1:

        author_known_defects[
            author
        ] += 1


    prod_files = (
        pr_to_production_files.get(
            pr_number,
            set(),
        )
    )


    for filename in prod_files:

        file_known_labels[
            filename
        ] += 1


        if outcome == 1:

            file_known_defects[
                filename
            ] += 1


# ============================================================
# CHRONOLOGICAL EVENTS
# ============================================================

events = (
    full_prs
    .sort_values(
        [
            "merged_at",
            "pr_number",
        ]
    )
    .reset_index(
        drop=True
    )
)


rows = []

total_events = len(
    events
)


for position, row in events.iterrows():

    pr_number = int(
        row["pr_number"]
    )

    current_time = (
        row["merged_at"]
    )


    if pd.isna(
        current_time
    ):
        continue


    # ========================================================
    # REVEAL TARGETS THAT ARE ALREADY KNOWABLE
    # ========================================================

    while (
        outcome_pointer
        < len(outcomes)
        and
        outcomes.loc[
            outcome_pointer,
            "observation_end",
        ]
        <= current_time
    ):

        outcome_row = (
            outcomes.loc[
                outcome_pointer
            ]
        )


        reveal_outcome(
            int(
                outcome_row[
                    "pr_number"
                ]
            ),
            int(
                outcome_row[
                    "observed_defect_90d"
                ]
            ),
        )


        outcome_pointer += 1


    author = author_key(
        row["author"],
        pr_number,
    )


    prod_files = (
        pr_to_production_files.get(
            pr_number,
            set(),
        )
    )


    # ========================================================
    # BUILD FEATURES BEFORE UPDATING HISTORY
    # ========================================================

    if pr_number in eligible_ids:

        # ----------------------------------------------------
        # AUTHOR FEATURES
        # ----------------------------------------------------

        prior_prs = (
            author_prior_prs[
                author
            ]
        )


        prior_prod_prs = (
            author_prior_prod_prs[
                author
            ]
        )


        prior_churn = (
            author_prior_code_churn[
                author
            ]
        )


        if author in author_first_merge:

            author_experience_days = (
                current_time
                -
                author_first_merge[
                    author
                ]
            ).total_seconds() / 86400

        else:

            author_experience_days = 0.0


        known_labels = (
            author_known_labels[
                author
            ]
        )


        known_defects = (
            author_known_defects[
                author
            ]
        )


        known_defect_rate = (
            safe_ratio(
                known_defects,
                known_labels,
            )
        )


        # ----------------------------------------------------
        # FILE FEATURES
        # ----------------------------------------------------

        prior_changes_values = [
            file_prior_changes[
                filename
            ]
            for filename
            in prod_files
        ]


        prior_authors_values = [
            len(
                file_prior_authors[
                    filename
                ]
            )
            for filename
            in prod_files
        ]


        known_file_labels_values = [
            file_known_labels[
                filename
            ]
            for filename
            in prod_files
        ]


        known_file_defects_values = [
            file_known_defects[
                filename
            ]
            for filename
            in prod_files
        ]


        known_file_rates = [

            safe_ratio(
                file_known_defects[
                    filename
                ],
                file_known_labels[
                    filename
                ],
            )

            for filename
            in prod_files
        ]


        if prod_files:

            unseen_files = sum(
                1
                for filename
                in prod_files
                if (
                    file_prior_changes[
                        filename
                    ] == 0
                )
            )


            file_unseen_ratio = (
                unseen_files
                / len(
                    prod_files
                )
            )

        else:

            file_unseen_ratio = 0.0


        rows.append(
            {
                "pr_number": (
                    pr_number
                ),

                # ============================================
                # AUTHOR HISTORY
                # ============================================

                "author_prior_prs": (
                    prior_prs
                ),

                "author_prior_prod_prs": (
                    prior_prod_prs
                ),

                "author_prior_code_churn": (
                    prior_churn
                ),

                "author_prior_avg_code_churn": (
                    safe_ratio(
                        prior_churn,
                        prior_prs,
                    )
                ),

                "author_experience_days": (
                    author_experience_days
                ),

                # Apenas targets cujo resultado
                # já era conhecido neste instante.

                "author_known_prior_labels": (
                    known_labels
                ),

                "author_known_prior_defects": (
                    known_defects
                ),

                "author_known_prior_defect_rate": (
                    known_defect_rate
                ),

                # ============================================
                # FILE HISTORY
                # ============================================

                "file_prior_changes_mean": (
                    mean_or_zero(
                        prior_changes_values
                    )
                ),

                "file_prior_changes_max": (
                    max_or_zero(
                        prior_changes_values
                    )
                ),

                "file_prior_authors_mean": (
                    mean_or_zero(
                        prior_authors_values
                    )
                ),

                "file_prior_authors_max": (
                    max_or_zero(
                        prior_authors_values
                    )
                ),

                "file_unseen_ratio": (
                    file_unseen_ratio
                ),

                # ============================================
                # KNOWN DEFECT HISTORY
                # ============================================

                "file_known_prior_labels_mean": (
                    mean_or_zero(
                        known_file_labels_values
                    )
                ),

                "file_known_prior_defects_mean": (
                    mean_or_zero(
                        known_file_defects_values
                    )
                ),

                "file_known_prior_defects_max": (
                    max_or_zero(
                        known_file_defects_values
                    )
                ),

                "file_known_prior_defect_rate_mean": (
                    mean_or_zero(
                        known_file_rates
                    )
                ),

                "file_known_prior_defect_rate_max": (
                    max_or_zero(
                        known_file_rates
                    )
                ),
            }
        )


    # ========================================================
    # UPDATE HISTORY ONLY AFTER FEATURE CALCULATION
    # ========================================================

    if author not in author_first_merge:

        author_first_merge[
            author
        ] = current_time


    author_prior_prs[
        author
    ] += 1


    author_prior_code_churn[
        author
    ] += float(
        row["code_churn"]
    )


    if prod_files:

        author_prior_prod_prs[
            author
        ] += 1


    for filename in prod_files:

        file_prior_changes[
            filename
        ] += 1

        file_prior_authors[
            filename
        ].add(
            author
        )


    if (
        position == 0
        or (position + 1) % 1000 == 0
        or position + 1 == total_events
    ):

        print(
            f"[{position + 1}/"
            f"{total_events}] "
            "eventos processados"
        )


# ============================================================
# HISTORICAL FEATURE DATAFRAME
# ============================================================

historical = pd.DataFrame(
    rows
)


# ============================================================
# MERGE WITH V1
# ============================================================

dataset = v1.merge(
    historical,
    on="pr_number",
    how="left",
    validate="one_to_one",
)


# ============================================================
# V2 FEATURE LIST
# ============================================================

v2_features = [
    "author_prior_prs",
    "author_prior_prod_prs",
    "author_prior_code_churn",
    "author_prior_avg_code_churn",
    "author_experience_days",

    "author_known_prior_labels",
    "author_known_prior_defects",
    "author_known_prior_defect_rate",

    "file_prior_changes_mean",
    "file_prior_changes_max",

    "file_prior_authors_mean",
    "file_prior_authors_max",

    "file_unseen_ratio",

    "file_known_prior_labels_mean",
    "file_known_prior_defects_mean",
    "file_known_prior_defects_max",

    "file_known_prior_defect_rate_mean",
    "file_known_prior_defect_rate_max",
]


# ============================================================
# VALIDATION
# ============================================================

print()
print("=" * 70)
print("VALIDAÇÃO V2")
print("=" * 70)


print(
    "Linhas:",
    len(dataset),
)

print(
    "PRs únicas:",
    dataset[
        "pr_number"
    ].nunique(),
)


duplicates = int(
    dataset
    .duplicated(
        subset="pr_number"
    )
    .sum()
)


missing_v2 = int(
    dataset[
        v2_features
    ]
    .isna()
    .sum()
    .sum()
)


negative_author_experience = int(
    (
        dataset[
            "author_experience_days"
        ] < 0
    ).sum()
)


invalid_author_rates = int(
    (
        (
            dataset[
                "author_known_prior_defect_rate"
            ] < 0
        )
        |
        (
            dataset[
                "author_known_prior_defect_rate"
            ] > 1
        )
    ).sum()
)


invalid_file_rates = int(
    (
        (
            dataset[
                "file_known_prior_defect_rate_mean"
            ] < 0
        )
        |
        (
            dataset[
                "file_known_prior_defect_rate_mean"
            ] > 1
        )
        |
        (
            dataset[
                "file_known_prior_defect_rate_max"
            ] < 0
        )
        |
        (
            dataset[
                "file_known_prior_defect_rate_max"
            ] > 1
        )
    ).sum()
)


invalid_unseen_ratio = int(
    (
        (
            dataset[
                "file_unseen_ratio"
            ] < 0
        )
        |
        (
            dataset[
                "file_unseen_ratio"
            ] > 1
        )
    ).sum()
)


print(
    "Duplicatas:",
    duplicates,
)

print(
    "Valores ausentes nas features V2:",
    missing_v2,
)

print(
    "Experiência negativa do autor:",
    negative_author_experience,
)

print(
    "Taxas históricas inválidas do autor:",
    invalid_author_rates,
)

print(
    "Taxas históricas inválidas dos arquivos:",
    invalid_file_rates,
)

print(
    "file_unseen_ratio inválido:",
    invalid_unseen_ratio,
)


# ============================================================
# LEAKAGE SANITY CHECK
# ============================================================

# Uma PR sem nenhum outcome histórico conhecido para o autor
# obrigatoriamente deve possuir zero defeitos históricos.

invalid_author_history = int(
    (
        (
            dataset[
                "author_known_prior_labels"
            ] == 0
        )
        &
        (
            dataset[
                "author_known_prior_defects"
            ] != 0
        )
    ).sum()
)


print(
    "Históricos impossíveis de autor:",
    invalid_author_history,
)


# ============================================================
# CRITICAL CHECKS
# ============================================================

critical = {
    "duplicatas": (
        duplicates
    ),

    "features V2 ausentes": (
        missing_v2
    ),

    "experiência negativa": (
        negative_author_experience
    ),

    "taxa de autor inválida": (
        invalid_author_rates
    ),

    "taxa de arquivo inválida": (
        invalid_file_rates
    ),

    "file_unseen_ratio inválido": (
        invalid_unseen_ratio
    ),

    "histórico impossível de autor": (
        invalid_author_history
    ),
}


failed = {
    key: value
    for key, value
    in critical.items()
    if value != 0
}


if len(dataset) != len(v1):

    failed[
        "quantidade de PRs"
    ] = (
        len(dataset)
        - len(v1)
    )


if failed:

    print()
    print("=" * 70)
    print("ERRO DE CONSISTÊNCIA")
    print("=" * 70)

    for key, value in (
        failed.items()
    ):

        print(
            f"{key}: {value}"
        )

    raise RuntimeError(
        "Feature Engineering V2 falhou "
        "nas verificações de consistência."
    )


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 70)
print("RESUMO DAS FEATURES HISTÓRICAS")
print("=" * 70)


print(
    dataset[
        v2_features
    ]
    .describe()
    .T[
        [
            "mean",
            "std",
            "min",
            "50%",
            "max",
        ]
    ]
    .round(3)
)


# ============================================================
# SAVE
# ============================================================

dataset.to_csv(
    OUTPUT_FILE,
    index=False,
)


print()
print("=" * 70)
print("FEATURE ENGINEERING V2 CONCLUÍDA")
print("=" * 70)


print(
    "Features V1:",
    len(
        [
            column
            for column in v1.columns
            if column not in [
                "pr_number",
                "collection_year",
                "observed_defect_90d",
            ]
        ]
    ),
)


print(
    "Novas features V2:",
    len(v2_features),
)


print(
    "Total de features:",
    len(
        [
            column
            for column in dataset.columns
            if column not in [
                "pr_number",
                "collection_year",
                "observed_defect_90d",
            ]
        ]
    ),
)


print(
    "PRs:",
    len(dataset),
)


print(
    "Positivos:",
    int(
        (
            dataset[
                "observed_defect_90d"
            ] == 1
        ).sum()
    ),
)


print(
    "Negativos:",
    int(
        (
            dataset[
                "observed_defect_90d"
            ] == 0
        ).sum()
    ),
)


print()

print(
    f"Dataset V2 salvo em:\n"
    f"{OUTPUT_FILE}"
)