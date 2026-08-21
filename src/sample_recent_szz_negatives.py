from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

EVIDENCE_FILE = (
    ROOT
    / "data"
    / "processed"
    / "pandas_szz_full_evidence.csv"
)

OUTPUT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "pandas_recent_szz_negative_sample.csv"
)


df = pd.read_csv(
    EVIDENCE_FILE
)


# 2024 e 2025
recent = df[
    df["collection_year"].isin(
        [2024, 2025]
    )
].copy()


# Apenas casos processados sem evidência SZZ.
negative = recent[
    (recent["high_confidence_szz"] == 0)
    &
    (recent["status"] != "error")
].copy()


# Um registro por par PR original × bug fix.
pairs = (
    negative
    .sort_values(
        [
            "collection_year",
            "days_after_merge",
        ]
    )
    .drop_duplicates(
        subset=[
            "pr_number",
            "bugfix_pr_number",
        ]
    )
)


# Amostra reproduzível:
# 5 pares de 2024 + 5 pares de 2025.
samples = []

for year in [2024, 2025]:

    year_df = pairs[
        pairs["collection_year"] == year
    ]

    n = min(
        5,
        len(year_df),
    )

    samples.append(
        year_df.sample(
            n=n,
            random_state=42,
        )
    )


sample = pd.concat(
    samples,
    ignore_index=True,
)


columns = [
    "pr_number",
    "original_title",
    "collection_year",
    "bugfix_pr_number",
    "bugfix_title",
    "days_after_merge",
    "filename",
    "original_commit",
    "bugfix_commit",
    "deleted_lines_checked",
    "status",
]


available_columns = [
    col
    for col in columns
    if col in sample.columns
]


sample = sample[
    available_columns
]


sample.to_csv(
    OUTPUT_FILE,
    index=False,
)


print()
print("=" * 70)
print("AMOSTRA DE NEGATIVOS RECENTES")
print("=" * 70)

print(
    sample.to_string(
        index=False
    )
)

print()
print(
    "Casos selecionados:",
    len(sample),
)

print()

print(
    f"Arquivo salvo em:\n{OUTPUT_FILE}"
)