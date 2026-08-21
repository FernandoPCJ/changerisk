from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

TARGET_FILE = (
    ROOT
    / "data"
    / "processed"
    / "pandas_targets_full.csv"
)

FILES_FILE = (
    ROOT
    / "data"
    / "raw"
    / "pandas_pr_files_full.csv"
)

OUTPUT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "pandas_ml_features_v1.csv"
)


# ============================================================
# RECONCILIAÇÕES CONHECIDAS
# ============================================================

# A PR #45983 apresenta inconsistência no endpoint resumido
# da GitHub API:
#
# changed_files = 0
# additions     = 0
# deletions     = 0
#
# A inspeção do merge commit via:
#
# git diff --numstat <parent> <merge_commit>
#
# confirmou:
#
# 4 arquivos
# 30 adições
# 2 deleções
#
# A quantidade de arquivos não precisa ser sobrescrita aqui,
# pois "changed_files" será derivada da lista reconciliada
# de arquivos da PR.
#
# Mantemos apenas as métricas de linhas que precisam de ajuste.

RECONCILED_CHANGE_METRICS = {
    45983: {
        "additions": 30,
        "deletions": 2,
    }
}


# ============================================================
# LOAD
# ============================================================

targets = pd.read_csv(
    TARGET_FILE
)

files = pd.read_csv(
    FILES_FILE
)


print()
print("=" * 70)
print("FEATURE ENGINEERING V1")
print("=" * 70)

print(
    "PRs recebidas:",
    len(targets),
)

print(
    "Registros de arquivos:",
    len(files),
)


# ============================================================
# BOOLEAN NORMALIZATION
# ============================================================

def normalize_bool(series):
    """
    Normaliza colunas booleanas que podem ter sido carregadas
    do CSV como bool ou string.
    """

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

files["is_test"] = normalize_bool(
    files["is_test"]
)

files["is_documentation"] = normalize_bool(
    files["is_documentation"]
)


# ============================================================
# FILE STATUS
# ============================================================

status = (
    files["status"]
    .fillna("")
    .astype(str)
    .str.strip()
)


files["file_added"] = (
    status == "A"
)

files["file_modified"] = (
    status == "M"
)

files["file_deleted"] = (
    status == "D"
)

files["file_renamed"] = (
    status.str.startswith("R")
)


# ============================================================
# OTHER FILE
# ============================================================

files["is_other"] = (
    ~files["is_production_code"]
    &
    ~files["is_test"]
    &
    ~files["is_documentation"]
)


# ============================================================
# AGGREGATE FILE FEATURES
# ============================================================

file_features = (
    files
    .groupby(
        "pr_number",
        as_index=False,
    )
    .agg(
        extracted_files=(
            "filename",
            "nunique",
        ),

        production_files_changed=(
            "is_production_code",
            "sum",
        ),

        test_files_changed=(
            "is_test",
            "sum",
        ),

        documentation_files_changed=(
            "is_documentation",
            "sum",
        ),

        other_files_changed=(
            "is_other",
            "sum",
        ),

        added_files=(
            "file_added",
            "sum",
        ),

        modified_files=(
            "file_modified",
            "sum",
        ),

        deleted_files=(
            "file_deleted",
            "sum",
        ),

        renamed_files=(
            "file_renamed",
            "sum",
        ),
    )
)


# ============================================================
# MERGE WITH TARGETS
# ============================================================

df = targets.merge(
    file_features,
    on="pr_number",
    how="left",
)


file_columns = [
    "extracted_files",
    "production_files_changed",
    "test_files_changed",
    "documentation_files_changed",
    "other_files_changed",
    "added_files",
    "modified_files",
    "deleted_files",
    "renamed_files",
]


df[file_columns] = (
    df[file_columns]
    .fillna(0)
    .astype(int)
)


# ============================================================
# BASIC NUMERIC NORMALIZATION
# ============================================================

numeric_base = [
    "commits",
    "changed_files",
    "additions",
    "deletions",
    "code_churn",
]


for column in numeric_base:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce",
    )


# ============================================================
# PRESERVE RAW API VALUES
# ============================================================

# Esses valores não entrarão no modelo.
# São preservados temporariamente apenas para auditoria.

df["changed_files_api"] = (
    df["changed_files"]
    .copy()
)

df["additions_api"] = (
    df["additions"]
    .copy()
)

df["deletions_api"] = (
    df["deletions"]
    .copy()
)

df["code_churn_api"] = (
    df["code_churn"]
    .copy()
)


# ============================================================
# RECONCILED FILE COUNT
# ============================================================

# Para Feature Engineering, utilizamos a quantidade derivada
# da lista reconciliada de arquivos da própria PR.
#
# Essa lista foi construída usando Git local e fallback /
# reconciliação com a API /pulls/{pr}/files quando necessário.

df["changed_files"] = (
    df["extracted_files"]
    .astype(int)
)


# ============================================================
# RECONCILED CHANGE METRICS
# ============================================================

for pr_number, metrics in (
    RECONCILED_CHANGE_METRICS.items()
):

    mask = (
        df["pr_number"]
        == pr_number
    )

    if not mask.any():

        raise RuntimeError(
            f"PR reconciliada #{pr_number} "
            "não foi encontrada no dataset."
        )

    df.loc[
        mask,
        "additions",
    ] = metrics["additions"]

    df.loc[
        mask,
        "deletions",
    ] = metrics["deletions"]


# ============================================================
# CODE CHURN
# ============================================================

# Code churn é definido como a soma de adições e deleções.
# Recalculamos para garantir consistência após reconciliações.

df["code_churn"] = (
    df["additions"]
    + df["deletions"]
)


# ============================================================
# DATES
# ============================================================

df["created_at"] = pd.to_datetime(
    df["created_at"],
    utc=True,
    errors="coerce",
)

df["merged_at"] = pd.to_datetime(
    df["merged_at"],
    utc=True,
    errors="coerce",
)


df["pr_duration_hours"] = (
    (
        df["merged_at"]
        - df["created_at"]
    )
    .dt.total_seconds()
    / 3600
)


# ============================================================
# SAFE DIVISION
# ============================================================

def safe_divide(
    numerator,
    denominator,
):
    """
    Realiza divisão evitando infinito e divisão por zero.
    """

    denominator = denominator.replace(
        0,
        np.nan,
    )

    result = (
        numerator
        / denominator
    )

    return (
        result
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .fillna(0)
    )


# ============================================================
# DERIVED FEATURES
# ============================================================

df["churn_per_file"] = safe_divide(
    df["code_churn"],
    df["changed_files"],
)


df["additions_per_file"] = safe_divide(
    df["additions"],
    df["changed_files"],
)


df["deletions_per_file"] = safe_divide(
    df["deletions"],
    df["changed_files"],
)


df["commits_per_file"] = safe_divide(
    df["commits"],
    df["changed_files"],
)


df["production_file_ratio"] = safe_divide(
    df["production_files_changed"],
    df["extracted_files"],
)


df["test_file_ratio"] = safe_divide(
    df["test_files_changed"],
    df["extracted_files"],
)


df["test_to_production_ratio"] = safe_divide(
    df["test_files_changed"],
    df["production_files_changed"],
)


df["addition_ratio"] = safe_divide(
    df["additions"],
    df["code_churn"],
)


df["deletion_ratio"] = safe_divide(
    df["deletions"],
    df["code_churn"],
)


# ============================================================
# BINARY FEATURES
# ============================================================

df["touches_tests"] = (
    df["test_files_changed"] > 0
).astype(int)


df["touches_documentation"] = (
    df[
        "documentation_files_changed"
    ] > 0
).astype(int)


df["has_file_rename"] = (
    df["renamed_files"] > 0
).astype(int)


# ============================================================
# TARGET
# ============================================================

df[
    "observed_defect_90d"
] = pd.to_numeric(
    df["observed_defect_90d"],
    errors="coerce",
).astype("Int64")


# ============================================================
# FEATURE SET
# ============================================================

feature_columns = [

    # --------------------------------------------------------
    # Identificação / metadados
    # --------------------------------------------------------

    "pr_number",
    "collection_year",

    # --------------------------------------------------------
    # Tamanho da mudança
    # --------------------------------------------------------

    "commits",
    "changed_files",
    "additions",
    "deletions",
    "code_churn",
    "pr_duration_hours",

    # --------------------------------------------------------
    # Arquivos modificados
    # --------------------------------------------------------

    "production_files_changed",
    "test_files_changed",
    "documentation_files_changed",
    "other_files_changed",

    "added_files",
    "modified_files",
    "deleted_files",
    "renamed_files",

    # --------------------------------------------------------
    # Razões
    # --------------------------------------------------------

    "churn_per_file",
    "additions_per_file",
    "deletions_per_file",
    "commits_per_file",

    "production_file_ratio",
    "test_file_ratio",
    "test_to_production_ratio",

    "addition_ratio",
    "deletion_ratio",

    # --------------------------------------------------------
    # Indicadores binários
    # --------------------------------------------------------

    "touches_tests",
    "touches_documentation",
    "has_file_rename",

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

    "observed_defect_90d",
]


dataset = df[
    feature_columns
].copy()


# ============================================================
# FEATURE LIST
# ============================================================

feature_only = [
    column
    for column in feature_columns
    if column not in [
        "pr_number",
        "collection_year",
        "observed_defect_90d",
    ]
]


# ============================================================
# VALIDATIONS
# ============================================================

print()
print("=" * 70)
print("VALIDAÇÃO")
print("=" * 70)


row_count = len(
    dataset
)

unique_prs = (
    dataset[
        "pr_number"
    ].nunique()
)

duplicates = (
    dataset
    .duplicated(
        subset="pr_number"
    )
    .sum()
)

missing_targets = (
    dataset[
        "observed_defect_90d"
    ]
    .isna()
    .sum()
)

negative_durations = int(
    (
        dataset[
            "pr_duration_hours"
        ] < 0
    ).sum()
)

missing_features = int(
    dataset[
        feature_only
    ]
    .isna()
    .sum()
    .sum()
)

zero_changed_files = int(
    (
        dataset[
            "changed_files"
        ] == 0
    ).sum()
)

no_production_files = int(
    (
        dataset[
            "production_files_changed"
        ] < 1
    ).sum()
)

churn_inconsistencies = int(
    (
        dataset[
            "code_churn"
        ]
        !=
        (
            dataset[
                "additions"
            ]
            +
            dataset[
                "deletions"
            ]
        )
    ).sum()
)


invalid_targets = int(
    (
        ~dataset[
            "observed_defect_90d"
        ].isin(
            [0, 1]
        )
    ).sum()
)


api_file_discrepancies = int(
    (
        df[
            "changed_files_api"
        ]
        !=
        df[
            "changed_files"
        ]
    ).sum()
)


print(
    "Linhas:",
    row_count,
)

print(
    "PRs únicas:",
    unique_prs,
)

print(
    "Duplicatas:",
    duplicates,
)

print(
    "Targets ausentes:",
    missing_targets,
)

print(
    "Targets inválidos:",
    invalid_targets,
)

print(
    "Durações negativas:",
    negative_durations,
)

print(
    "Valores ausentes nas features:",
    missing_features,
)

print(
    "PRs com changed_files = 0:",
    zero_changed_files,
)

print(
    "PRs sem arquivo de produção:",
    no_production_files,
)

print(
    "Inconsistências code_churn != additions + deletions:",
    churn_inconsistencies,
)

print(
    "Divergências preservadas API x arquivos reconciliados:",
    api_file_discrepancies,
)


# ============================================================
# VALIDATE KNOWN RECONCILIATION
# ============================================================

pr_45983 = dataset[
    dataset[
        "pr_number"
    ] == 45983
]


if len(pr_45983) != 1:

    raise RuntimeError(
        "A PR #45983 deveria aparecer exatamente "
        "uma vez no dataset."
    )


pr_45983 = pr_45983.iloc[0]


expected_45983 = {
    "changed_files": 4,
    "additions": 30,
    "deletions": 2,
    "code_churn": 32,
}


for column, expected_value in (
    expected_45983.items()
):

    actual_value = (
        pr_45983[column]
    )

    if actual_value != expected_value:

        raise RuntimeError(
            f"Reconciliação da PR #45983 inválida: "
            f"{column}={actual_value}; "
            f"esperado={expected_value}"
        )


print()
print(
    "PR #45983 reconciliada corretamente:"
)

print(
    "changed_files =",
    int(
        pr_45983[
            "changed_files"
        ]
    ),
)

print(
    "additions =",
    int(
        pr_45983[
            "additions"
        ]
    ),
)

print(
    "deletions =",
    int(
        pr_45983[
            "deletions"
        ]
    ),
)

print(
    "code_churn =",
    int(
        pr_45983[
            "code_churn"
        ]
    ),
)


# ============================================================
# CRITICAL VALIDATION
# ============================================================

critical_errors = {
    "duplicatas": (
        int(duplicates)
    ),

    "targets ausentes": (
        int(missing_targets)
    ),

    "targets inválidos": (
        int(invalid_targets)
    ),

    "durações negativas": (
        int(negative_durations)
    ),

    "features ausentes": (
        int(missing_features)
    ),

    "changed_files = 0": (
        int(zero_changed_files)
    ),

    "PRs sem código de produção": (
        int(no_production_files)
    ),

    "code churn inconsistente": (
        int(churn_inconsistencies)
    ),
}


failed_checks = {
    name: value
    for name, value
    in critical_errors.items()
    if value != 0
}


if failed_checks:

    print()
    print("=" * 70)
    print("ERRO DE CONSISTÊNCIA")
    print("=" * 70)

    for name, value in (
        failed_checks.items()
    ):

        print(
            f"{name}: {value}"
        )

    raise RuntimeError(
        "O dataset não foi salvo porque "
        "existem inconsistências críticas."
    )


# ============================================================
# TARGET DISTRIBUTION
# ============================================================

print()
print("=" * 70)
print("TARGET")
print("=" * 70)


target_counts = (
    dataset[
        "observed_defect_90d"
    ]
    .value_counts()
    .sort_index()
)


print(
    target_counts
)


positive_rate = (
    (
        dataset[
            "observed_defect_90d"
        ] == 1
    )
    .mean()
    * 100
)


print()

print(
    "Taxa positiva:",
    round(
        positive_rate,
        2,
    ),
    "%",
)


# ============================================================
# TARGET BY YEAR
# ============================================================

print()
print("=" * 70)
print("TARGET POR ANO")
print("=" * 70)


target_year = (
    dataset
    .groupby(
        [
            "collection_year",
            "observed_defect_90d",
        ]
    )
    .size()
    .unstack(
        fill_value=0
    )
)


print(
    target_year
)


# ============================================================
# FEATURE SUMMARY
# ============================================================

print()
print("=" * 70)
print("RESUMO DAS FEATURES")
print("=" * 70)


print(
    dataset[
        feature_only
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
    .round(2)
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
print("FEATURE ENGINEERING V1 CONCLUÍDA")
print("=" * 70)


print(
    "Número de features:",
    len(feature_only),
)

print(
    "Número de PRs:",
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
    f"Dataset salvo em:\n"
    f"{OUTPUT_FILE}"
)