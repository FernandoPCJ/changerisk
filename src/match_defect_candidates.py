import pandas as pd


ORIGINAL_PRS_FILE = "data/raw/pandas_pulls_pilot_100_enriched.csv"
ORIGINAL_FILES_FILE = "data/raw/pandas_pilot_pr_files.csv"

BUGFIX_PRS_FILE = "data/raw/pandas_bugfix_pulls_enriched.csv"
BUGFIX_FILES_FILE = "data/raw/pandas_bugfix_pr_files.csv"

DETAIL_OUTPUT = "data/processed/pandas_defect_candidates_detail.csv"
SUMMARY_OUTPUT = "data/processed/pandas_defect_candidates_summary.csv"


# ==========================================
# LOAD
# ==========================================

original_prs = pd.read_csv(ORIGINAL_PRS_FILE)
original_files = pd.read_csv(ORIGINAL_FILES_FILE)

bugfix_prs = pd.read_csv(BUGFIX_PRS_FILE)
bugfix_files = pd.read_csv(BUGFIX_FILES_FILE)

def is_production_file(filename):
    return (
        filename.startswith("pandas/")
        and not filename.startswith("pandas/tests/")
    )


original_files = original_files[
    original_files["filename"].apply(is_production_file)
].copy()

bugfix_files = bugfix_files[
    bugfix_files["filename"].apply(is_production_file)
].copy()


# ==========================================
# DATES
# ==========================================

original_prs["merged_at"] = pd.to_datetime(
    original_prs["merged_at"],
    utc=True
)

bugfix_prs["merged_at"] = pd.to_datetime(
    bugfix_prs["merged_at"],
    utc=True
)

bugfix_files["bugfix_merged_at"] = pd.to_datetime(
    bugfix_files["bugfix_merged_at"],
    utc=True
)


# ==========================================
# ORIGINAL PR + FILES
# ==========================================

original = original_files.merge(
    original_prs[
        [
            "pr_number",
            "merged_at",
            "title",
        ]
    ],
    on="pr_number",
    how="left",
)

original = original.rename(
    columns={
        "merged_at": "original_merged_at",
        "title": "original_title",
    }
)


# ==========================================
# BUGFIX PR + FILES
# ==========================================

bugfix = bugfix_files.merge(
    bugfix_prs[
        [
            "pr_number",
            "title",
        ]
    ],
    left_on="bugfix_pr_number",
    right_on="pr_number",
    how="left",
)

bugfix = bugfix.rename(
    columns={
        "title": "bugfix_title",
    }
)

bugfix = bugfix.drop(
    columns=["pr_number"]
)


# ==========================================
# MATCH BY SAME FILE
# ==========================================

candidates = original.merge(
    bugfix,
    on="filename",
    how="inner",
    suffixes=("_original", "_bugfix"),
)


# ==========================================
# TEMPORAL RELATION
# ==========================================

candidates["days_after_merge"] = (
    candidates["bugfix_merged_at"]
    - candidates["original_merged_at"]
).dt.total_seconds() / 86400


candidates = candidates[
    (candidates["days_after_merge"] > 0)
    & (candidates["days_after_merge"] <= 90)
    & (
        candidates["pr_number"]
        != candidates["bugfix_pr_number"]
    )
].copy()


# ==========================================
# CLEAN DETAIL
# ==========================================

detail = candidates[
    [
        "pr_number",
        "original_title",
        "original_merged_at",
        "bugfix_pr_number",
        "bugfix_title",
        "bugfix_merged_at",
        "days_after_merge",
        "filename",
    ]
].copy()

detail = detail.sort_values(
    [
        "pr_number",
        "days_after_merge",
        "bugfix_pr_number",
    ]
)

detail.to_csv(
    DETAIL_OUTPUT,
    index=False,
)


# ==========================================
# SUMMARY BY PR PAIR
# ==========================================

summary = (
    detail.groupby(
        [
            "pr_number",
            "original_title",
            "original_merged_at",
            "bugfix_pr_number",
            "bugfix_title",
            "bugfix_merged_at",
            "days_after_merge",
        ],
        as_index=False,
    )
    .agg(
        shared_files=("filename", "nunique")
    )
)

summary = summary.sort_values(
    [
        "pr_number",
        "days_after_merge",
        "shared_files",
    ],
    ascending=[True, True, False],
)

summary.to_csv(
    SUMMARY_OUTPUT,
    index=False,
)


# ==========================================
# RESULTS
# ==========================================

original_with_candidates = (
    summary["pr_number"].nunique()
    if not summary.empty
    else 0
)

print()
print("Cruzamento concluído.")

print(
    "PRs originais analisadas:",
    original_prs["pr_number"].nunique()
)

print(
    "PRs originais com pelo menos um candidato:",
    original_with_candidates
)

print(
    "Pares PR original × Bug fix:",
    len(summary)
)

print(
    "Correspondências detalhadas de arquivos:",
    len(detail)
)

if not summary.empty:

    print()
    print("Primeiros candidatos:")

    print(
        summary[
            [
                "pr_number",
                "bugfix_pr_number",
                "days_after_merge",
                "shared_files",
            ]
        ].head(20)
    )

print()
print(
    f"Detalhes salvos em {DETAIL_OUTPUT}"
)

print(
    f"Resumo salvo em {SUMMARY_OUTPUT}"
)