from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

V1_FILE = (
    ROOT
    / "data"
    / "processed"
    / "pandas_ml_features_v1.csv"
)

V2_FILE = (
    ROOT
    / "data"
    / "processed"
    / "pandas_ml_features_v2.csv"
)

STRUCTURAL_OUTPUT = (
    ROOT
    / "data"
    / "processed"
    / "pandas_ml_structural.csv"
)

EXTENDED_OUTPUT = (
    ROOT
    / "data"
    / "processed"
    / "pandas_ml_extended.csv"
)

STABLE_OUTPUT = (
    ROOT
    / "data"
    / "processed"
    / "pandas_ml_stable.csv"
)

MANIFEST_OUTPUT = (
    ROOT
    / "data"
    / "processed"
    / "pandas_ml_feature_manifest.csv"
)


# ============================================================
# LOAD
# ============================================================

v1 = pd.read_csv(
    V1_FILE
)

v2 = pd.read_csv(
    V2_FILE
)


# ============================================================
# FIXED COLUMNS
# ============================================================

ID_COLUMNS = [
    "pr_number",
    "collection_year",
]

TARGET = (
    "observed_defect_90d"
)


# ============================================================
# DETERMINISTIC REDUNDANCIES
# ============================================================

DROP_REDUNDANT = [
    "code_churn",
    "deletion_ratio",
    "touches_tests",
    "touches_documentation",
    "has_file_rename",
]


# ============================================================
# TEMPORAL DRIFT
# ============================================================

TEMPORAL_DRIFT_FEATURES = [
    "file_known_prior_labels_mean",
    "file_prior_authors_mean",
    "file_prior_changes_mean",
    "file_known_prior_defects_mean",
    "file_prior_changes_max",
    "file_prior_authors_max",
]


# ============================================================
# ORIGINAL FEATURE SETS
# ============================================================

v1_features = [
    column
    for column in v1.columns
    if column not in (
        ID_COLUMNS
        + [TARGET]
    )
]


v2_features = [
    column
    for column in v2.columns
    if column not in (
        ID_COLUMNS
        + [TARGET]
    )
]


# ============================================================
# STRUCTURAL FEATURE SET
# ============================================================

structural_features = [
    feature
    for feature in v1_features
    if feature not in DROP_REDUNDANT
]


# ============================================================
# EXTENDED FEATURE SET
# ============================================================

extended_features = [
    feature
    for feature in v2_features
    if feature not in DROP_REDUNDANT
]


# ============================================================
# STABLE FEATURE SET
# ============================================================

stable_features = [
    feature
    for feature in extended_features
    if feature not in TEMPORAL_DRIFT_FEATURES
]


# ============================================================
# BUILD DATASETS
# ============================================================

structural = v2[
    ID_COLUMNS
    + structural_features
    + [TARGET]
].copy()


extended = v2[
    ID_COLUMNS
    + extended_features
    + [TARGET]
].copy()


stable = v2[
    ID_COLUMNS
    + stable_features
    + [TARGET]
].copy()


# ============================================================
# VALIDATION FUNCTION
# ============================================================

def validate_dataset(
    name,
    dataset,
    features,
):

    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    rows = len(
        dataset
    )

    unique_prs = (
        dataset[
            "pr_number"
        ]
        .nunique()
    )

    duplicates = int(
        dataset
        .duplicated(
            subset="pr_number"
        )
        .sum()
    )

    missing = int(
        dataset[
            features
        ]
        .isna()
        .sum()
        .sum()
    )

    positives = int(
        (
            dataset[
                TARGET
            ] == 1
        ).sum()
    )

    negatives = int(
        (
            dataset[
                TARGET
            ] == 0
        ).sum()
    )

    print(
        "PRs:",
        rows,
    )

    print(
        "PRs únicas:",
        unique_prs,
    )

    print(
        "Features:",
        len(features),
    )

    print(
        "Duplicatas:",
        duplicates,
    )

    print(
        "Valores ausentes:",
        missing,
    )

    print(
        "Positivos:",
        positives,
    )

    print(
        "Negativos:",
        negatives,
    )


    if rows != 6822:
        raise RuntimeError(
            f"{name}: quantidade "
            "inesperada de PRs."
        )


    if unique_prs != rows:
        raise RuntimeError(
            f"{name}: existem PRs duplicadas."
        )


    if duplicates != 0:
        raise RuntimeError(
            f"{name}: duplicatas encontradas."
        )


    if missing != 0:
        raise RuntimeError(
            f"{name}: valores ausentes."
        )


    if positives != 114:

        raise RuntimeError(
            f"{name}: quantidade de "
            "positivos inesperada."
        )


    if negatives != 6708:

        raise RuntimeError(
            f"{name}: quantidade de "
            "negativos inesperada."
        )


# ============================================================
# VALIDATE
# ============================================================

validate_dataset(
    "DATASET STRUCTURAL",
    structural,
    structural_features,
)

validate_dataset(
    "DATASET EXTENDED",
    extended,
    extended_features,
)

validate_dataset(
    "DATASET STABLE",
    stable,
    stable_features,
)


# ============================================================
# FEATURE MANIFEST
# ============================================================

manifest_rows = []


for feature in v2_features:

    if feature in DROP_REDUNDANT:

        status = (
            "removed_deterministic_redundancy"
        )

    elif feature in TEMPORAL_DRIFT_FEATURES:

        status = (
            "temporal_drift_flag"
        )

    else:

        status = (
            "retained"
        )


    if feature in v1_features:

        group = (
            "structural"
        )

    else:

        group = (
            "historical"
        )


    manifest_rows.append(
        {
            "feature": (
                feature
            ),

            "group": (
                group
            ),

            "status": (
                status
            ),

            "in_structural": (
                feature
                in structural_features
            ),

            "in_extended": (
                feature
                in extended_features
            ),

            "in_stable": (
                feature
                in stable_features
            ),
        }
    )


manifest = pd.DataFrame(
    manifest_rows
)


# ============================================================
# SAVE
# ============================================================

structural.to_csv(
    STRUCTURAL_OUTPUT,
    index=False,
)

extended.to_csv(
    EXTENDED_OUTPUT,
    index=False,
)

stable.to_csv(
    STABLE_OUTPUT,
    index=False,
)

manifest.to_csv(
    MANIFEST_OUTPUT,
    index=False,
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("DATASETS DE ML CONSTRUÍDOS")
print("=" * 70)

print(
    "Structural:",
    len(
        structural_features
    ),
    "features",
)

print(
    "Extended:",
    len(
        extended_features
    ),
    "features",
)

print(
    "Stable:",
    len(
        stable_features
    ),
    "features",
)

print()

print(
    "Redundâncias determinísticas removidas:"
)

for feature in DROP_REDUNDANT:

    print(
        " -",
        feature,
    )


print()

print(
    "Features sinalizadas por drift temporal:"
)

for feature in TEMPORAL_DRIFT_FEATURES:

    print(
        " -",
        feature,
    )


print()
print(
    f"Structural:\n"
    f"{STRUCTURAL_OUTPUT}"
)

print()
print(
    f"Extended:\n"
    f"{EXTENDED_OUTPUT}"
)

print()
print(
    f"Stable:\n"
    f"{STABLE_OUTPUT}"
)

print()
print(
    f"Manifest:\n"
    f"{MANIFEST_OUTPUT}"
)