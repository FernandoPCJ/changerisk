from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

PRS_FILE = (
    ROOT / "data" / "raw" / "pandas_pulls_full_enriched.csv"
)

FILES_FILE = (
    ROOT / "data" / "raw" / "pandas_pr_files_full.csv"
)

OUTPUT_FILE = (
    ROOT / "data" / "processed" / "pandas_eligible_population.csv"
)


# ============================================================
# LOAD
# ============================================================

prs = pd.read_csv(
    PRS_FILE
)

files = pd.read_csv(
    FILES_FILE
)


# ============================================================
# CLASSIFICAR PRs
# ============================================================

production_prs = set(
    files.loc[
        files["is_production_code"] == True,
        "pr_number",
    ].astype(int)
)

test_prs = set(
    files.loc[
        files["is_test"] == True,
        "pr_number",
    ].astype(int)
)

documentation_prs = set(
    files.loc[
        files["is_documentation"] == True,
        "pr_number",
    ].astype(int)
)


population = prs.copy()


population["touches_production_code"] = (
    population["pr_number"]
    .isin(production_prs)
    .astype(int)
)

population["touches_tests"] = (
    population["pr_number"]
    .isin(test_prs)
    .astype(int)
)

population["touches_documentation"] = (
    population["pr_number"]
    .isin(documentation_prs)
    .astype(int)
)


# ============================================================
# JANELA TEMPORAL
# ============================================================

population["merged_at"] = pd.to_datetime(
    population["merged_at"],
    utc=True,
)

population["observation_end"] = (
    population["merged_at"]
    + pd.Timedelta(days=90)
)

OBSERVATION_CUTOFF = pd.Timestamp(
    "2026-03-31T23:59:59Z"
)

population["observation_window_complete"] = (
    population["observation_end"]
    <= OBSERVATION_CUTOFF
).astype(int)


# ============================================================
# ELEGIBILIDADE
# ============================================================

population["target_eligible"] = (
    (
        population["touches_production_code"] == 1
    )
    &
    (
        population["observation_window_complete"] == 1
    )
).astype(int)


eligible = population[
    population["target_eligible"] == 1
].copy()


eligible.to_csv(
    OUTPUT_FILE,
    index=False,
)


# ============================================================
# RESUMO
# ============================================================

print()
print("=" * 60)
print("POPULAÇÃO ELEGÍVEL")
print("=" * 60)

print(
    "PRs totais:",
    len(population),
)

print(
    "PRs que alteram código de produção:",
    int(
        population[
            "touches_production_code"
        ].sum()
    ),
)

print(
    "PRs que alteram testes:",
    int(
        population[
            "touches_tests"
        ].sum()
    ),
)

print(
    "PRs que alteram documentação:",
    int(
        population[
            "touches_documentation"
        ].sum()
    ),
)

print(
    "PRs com janela de 90 dias completa:",
    int(
        population[
            "observation_window_complete"
        ].sum()
    ),
)

print(
    "PRs elegíveis para o target:",
    len(eligible),
)

print()
print("Elegíveis por ano:")

print(
    eligible[
        "collection_year"
    ]
    .value_counts()
    .sort_index()
)

print()

print(
    f"Arquivo salvo em:\n"
    f"{OUTPUT_FILE}"
)