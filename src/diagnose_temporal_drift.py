from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import ks_2samp


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
    / "pandas_temporal_drift_diagnostics.csv"
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


development = (
    df[
        df["collection_year"].isin(
            [2022, 2023]
        )
    ]
    .copy()
)


future = (
    df[
        df["collection_year"].isin(
            [2024, 2025]
        )
    ]
    .copy()
)


print()
print("=" * 76)
print("DIAGNÓSTICO DE DRIFT TEMPORAL")
print("=" * 76)

print(
    "Desenvolvimento 2022-2023:",
    len(development),
)

print(
    "Futuro 2024-2025:",
    len(future),
)


# ============================================================
# PSI
# ============================================================

def calculate_psi(
    expected,
    actual,
    bins=10,
):
    """
    Population Stability Index.

    Os limites são definidos usando a distribuição
    do período de desenvolvimento.
    """

    expected = pd.to_numeric(
        expected,
        errors="coerce",
    ).dropna().to_numpy()

    actual = pd.to_numeric(
        actual,
        errors="coerce",
    ).dropna().to_numpy()


    if (
        len(expected) == 0
        or len(actual) == 0
    ):
        return np.nan


    # Quantis do desenvolvimento.
    boundaries = np.unique(
        np.quantile(
            expected,
            np.linspace(
                0,
                1,
                bins + 1,
            ),
        )
    )


    # Feature quase constante.
    if len(boundaries) < 3:
        return 0.0


    boundaries[0] = -np.inf
    boundaries[-1] = np.inf


    expected_counts, _ = np.histogram(
        expected,
        bins=boundaries,
    )

    actual_counts, _ = np.histogram(
        actual,
        bins=boundaries,
    )


    expected_pct = (
        expected_counts
        / expected_counts.sum()
    )

    actual_pct = (
        actual_counts
        / actual_counts.sum()
    )


    epsilon = 1e-6


    expected_pct = np.clip(
        expected_pct,
        epsilon,
        None,
    )

    actual_pct = np.clip(
        actual_pct,
        epsilon,
        None,
    )


    psi = np.sum(
        (
            actual_pct
            - expected_pct
        )
        *
        np.log(
            actual_pct
            / expected_pct
        )
    )


    return float(
        psi
    )


# ============================================================
# FEATURE DRIFT
# ============================================================

rows = []


for feature in features:

    dev = pd.to_numeric(
        development[
            feature
        ],
        errors="coerce",
    ).dropna()


    fut = pd.to_numeric(
        future[
            feature
        ],
        errors="coerce",
    ).dropna()


    ks = ks_2samp(
        dev,
        fut,
    )


    psi = calculate_psi(
        dev,
        fut,
    )


    dev_median = (
        dev.median()
    )

    future_median = (
        fut.median()
    )


    rows.append(
        {
            "feature": feature,

            "development_mean": (
                dev.mean()
            ),

            "future_mean": (
                fut.mean()
            ),

            "development_median": (
                dev_median
            ),

            "future_median": (
                future_median
            ),

            "median_difference": (
                future_median
                - dev_median
            ),

            "ks_statistic": (
                ks.statistic
            ),

            "ks_pvalue": (
                ks.pvalue
            ),

            "psi": (
                psi
            ),
        }
    )


results = pd.DataFrame(
    rows
)


# ============================================================
# DRIFT FLAGS
# ============================================================

results[
    "psi_level"
] = pd.cut(
    results[
        "psi"
    ],
    bins=[
        -np.inf,
        0.10,
        0.25,
        np.inf,
    ],
    labels=[
        "low",
        "moderate",
        "high",
    ],
)


results[
    "ks_flag"
] = (
    results[
        "ks_statistic"
    ] >= 0.10
)


results = (
    results
    .sort_values(
        [
            "psi",
            "ks_statistic",
        ],
        ascending=False,
    )
    .reset_index(drop=True)
)


# ============================================================
# TARGET DRIFT
# ============================================================

development_prevalence = (
    development[
        TARGET
    ].mean()
)


future_prevalence = (
    future[
        TARGET
    ].mean()
)


prevalence_ratio = (
    future_prevalence
    / development_prevalence
)


print()
print("=" * 76)
print("DRIFT DO TARGET")
print("=" * 76)

print(
    "Prevalência desenvolvimento:",
    round(
        development_prevalence,
        6,
    ),
)

print(
    "Prevalência futuro:",
    round(
        future_prevalence,
        6,
    ),
)

print(
    "Razão futuro / desenvolvimento:",
    round(
        prevalence_ratio,
        4,
    ),
)


# ============================================================
# PRINT FEATURE DRIFT
# ============================================================

print()
print("=" * 76)
print("FEATURES COM MAIOR DRIFT")
print("=" * 76)


print(
    results[
        [
            "feature",
            "development_median",
            "future_median",
            "ks_statistic",
            "psi",
            "psi_level",
        ]
    ]
    .head(20)
    .round(4)
    .to_string(
        index=False
    )
)


print()
print(
    "Features com PSI >= 0.10:",
    int(
        (
            results[
                "psi"
            ] >= 0.10
        ).sum()
    ),
)


print(
    "Features com PSI >= 0.25:",
    int(
        (
            results[
                "psi"
            ] >= 0.25
        ).sum()
    ),
)


print(
    "Features com KS >= 0.10:",
    int(
        results[
            "ks_flag"
        ].sum()
    ),
)


# ============================================================
# SAVE
# ============================================================

results.to_csv(
    OUTPUT_FILE,
    index=False,
)


print()
print("=" * 76)
print("DIAGNÓSTICO DE DRIFT CONCLUÍDO")
print("=" * 76)

print(
    f"Resultado salvo em:\n"
    f"{OUTPUT_FILE}"
)