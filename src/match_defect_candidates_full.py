from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

ELIGIBLE_FILE = (
    ROOT
    / "data"
    / "processed"
    / "pandas_eligible_population.csv"
)

ORIGINAL_FILES_FILE = (
    ROOT
    / "data"
    / "raw"
    / "pandas_pr_files_full.csv"
)

BUGFIX_PRS_FILE = (
    ROOT
    / "data"
    / "raw"
    / "pandas_bugfix_pulls_enriched.csv"
)

BUGFIX_FILES_FILE = (
    ROOT
    / "data"
    / "raw"
    / "pandas_bugfix_pr_files.csv"
)

DETAIL_OUTPUT = (
    ROOT
    / "data"
    / "processed"
    / "pandas_defect_candidates_full_detail.csv"
)

SUMMARY_OUTPUT = (
    ROOT
    / "data"
    / "processed"
    / "pandas_defect_candidates_full_summary.csv"
)


# ============================================================
# LOAD
# ============================================================

eligible = pd.read_csv(
    ELIGIBLE_FILE
)

original_files = pd.read_csv(
    ORIGINAL_FILES_FILE
)

bugfix_prs = pd.read_csv(
    BUGFIX_PRS_FILE
)

bugfix_files = pd.read_csv(
    BUGFIX_FILES_FILE
)


# ============================================================
# DATES
# ============================================================

eligible["merged_at"] = pd.to_datetime(
    eligible["merged_at"],
    utc=True,
)

bugfix_files["bugfix_merged_at"] = pd.to_datetime(
    bugfix_files["bugfix_merged_at"],
    utc=True,
)


# ============================================================
# PRODUCTION FILE FILTER
# ============================================================

def is_production_file(filename):
    return (
        filename.startswith("pandas/")
        and not filename.startswith(
            "pandas/tests/"
        )
    )


original_files = original_files[
    original_files[
        "filename"
    ].apply(
        is_production_file
    )
].copy()


bugfix_files = bugfix_files[
    bugfix_files[
        "filename"
    ].apply(
        is_production_file
    )
].copy()


# ============================================================
# KEEP ONLY ELIGIBLE ORIGINAL PRs
# ============================================================

eligible_ids = set(
    eligible[
        "pr_number"
    ].astype(int)
)


original_files = original_files[
    original_files[
        "pr_number"
    ].isin(
        eligible_ids
    )
].copy()


# ============================================================
# ORIGINAL PR + FILES
# ============================================================

# Usamos somente as colunas necessárias dos arquivos.
# Isso evita duplicação de collection_year durante o merge.

original = original_files[
    [
        "pr_number",
        "filename",
    ]
].merge(
    eligible[
        [
            "pr_number",
            "title",
            "merged_at",
            "collection_year",
        ]
    ],
    on="pr_number",
    how="inner",
)


original = original.rename(
    columns={
        "title": (
            "original_title"
        ),
        "merged_at": (
            "original_merged_at"
        ),
    }
)


# ============================================================
# BUGFIX PR + FILES
# ============================================================

# bugfix_files JÁ possui bugfix_merged_at.
# Portanto, buscamos somente o título no dataframe das PRs.

bugfix = bugfix_files[
    [
        "bugfix_pr_number",
        "bugfix_merged_at",
        "filename",
    ]
].merge(
    bugfix_prs[
        [
            "pr_number",
            "title",
        ]
    ],
    left_on="bugfix_pr_number",
    right_on="pr_number",
    how="inner",
)


bugfix = bugfix.rename(
    columns={
        "title": (
            "bugfix_title"
        ),
    }
)


bugfix = bugfix.drop(
    columns=[
        "pr_number"
    ]
)


# ============================================================
# VALIDATION BEFORE MATCH
# ============================================================

print()
print("PRs elegíveis:", len(eligible))

print(
    "Arquivos de produção das PRs elegíveis:",
    len(original),
)

print(
    "Arquivos de produção dos bug fixes:",
    len(bugfix),
)

print()


# ============================================================
# SAME PRODUCTION FILE
# ============================================================

print(
    "Cruzando arquivos de produção..."
)


candidates = original.merge(
    bugfix,
    on="filename",
    how="inner",
)


print(
    "Correspondências antes do filtro temporal:",
    len(candidates),
)


# ============================================================
# TEMPORAL WINDOW
# ============================================================

candidates[
    "days_after_merge"
] = (
    candidates[
        "bugfix_merged_at"
    ]
    -
    candidates[
        "original_merged_at"
    ]
).dt.total_seconds() / 86400


candidates = candidates[
    (
        candidates[
            "days_after_merge"
        ] > 0
    )
    &
    (
        candidates[
            "days_after_merge"
        ] <= 90
    )
    &
    (
        candidates[
            "pr_number"
        ]
        !=
        candidates[
            "bugfix_pr_number"
        ]
    )
].copy()


print(
    "Correspondências após janela de 90 dias:",
    len(candidates),
)


# ============================================================
# DETAIL
# ============================================================

detail = candidates[
    [
        "pr_number",
        "original_title",
        "original_merged_at",
        "collection_year",
        "bugfix_pr_number",
        "bugfix_title",
        "bugfix_merged_at",
        "days_after_merge",
        "filename",
    ]
].copy()


detail = (
    detail
    .drop_duplicates()
    .sort_values(
        [
            "pr_number",
            "days_after_merge",
            "bugfix_pr_number",
            "filename",
        ]
    )
    .reset_index(
        drop=True
    )
)


detail.to_csv(
    DETAIL_OUTPUT,
    index=False,
)


# ============================================================
# SUMMARY
# ============================================================

summary = (
    detail
    .groupby(
        [
            "pr_number",
            "original_title",
            "original_merged_at",
            "collection_year",
            "bugfix_pr_number",
            "bugfix_title",
            "bugfix_merged_at",
            "days_after_merge",
        ],
        as_index=False,
    )
    .agg(
        shared_files=(
            "filename",
            "nunique",
        )
    )
)


summary = (
    summary
    .sort_values(
        [
            "pr_number",
            "days_after_merge",
            "shared_files",
        ],
        ascending=[
            True,
            True,
            False,
        ],
    )
    .reset_index(
        drop=True
    )
)


summary.to_csv(
    SUMMARY_OUTPUT,
    index=False,
)


# ============================================================
# SUMMARY
# ============================================================

prs_with_candidates = (
    summary[
        "pr_number"
    ].nunique()
    if not summary.empty
    else 0
)


bugfixes_used = (
    summary[
        "bugfix_pr_number"
    ].nunique()
    if not summary.empty
    else 0
)


print()
print("=" * 60)
print("CANDIDATOS DEFINITIVOS")
print("=" * 60)


print(
    "PRs elegíveis:",
    len(eligible),
)


print(
    "PRs elegíveis com pelo menos um candidato:",
    prs_with_candidates,
)


print(
    "Pares PR original × Bug fix:",
    len(summary),
)


print(
    "Correspondências detalhadas de arquivos:",
    len(detail),
)


print(
    "Bug fixes diferentes utilizados:",
    bugfixes_used,
)


print()
print(
    "PRs com candidato por ano:"
)


if not summary.empty:

    candidate_prs = (
        summary[
            [
                "pr_number",
                "collection_year",
            ]
        ]
        .drop_duplicates(
            subset="pr_number"
        )
    )


    print(
        candidate_prs[
            "collection_year"
        ]
        .value_counts()
        .sort_index()
    )


print()

print(
    f"Detalhes salvos em:\n"
    f"{DETAIL_OUTPUT}"
)

print()

print(
    f"Resumo salvo em:\n"
    f"{SUMMARY_OUTPUT}"
)