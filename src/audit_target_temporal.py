from pathlib import Path
import subprocess

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

REPO_DIR = (
    ROOT
    / "repositories"
    / "pandas"
)

ELIGIBLE_FILE = (
    ROOT
    / "data"
    / "processed"
    / "pandas_eligible_population.csv"
)

CANDIDATES_FILE = (
    ROOT
    / "data"
    / "processed"
    / "pandas_defect_candidates_full_detail.csv"
)

EVIDENCE_FILE = (
    ROOT
    / "data"
    / "processed"
    / "pandas_szz_full_evidence.csv"
)

TARGET_FILE = (
    ROOT
    / "data"
    / "processed"
    / "pandas_targets_full.csv"
)

BUGFIX_FILE = (
    ROOT
    / "data"
    / "raw"
    / "pandas_bugfix_pulls_enriched.csv"
)


def run_git(args):
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_DIR,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    if result.returncode != 0:
        return None

    return result.stdout.strip()


eligible = pd.read_csv(
    ELIGIBLE_FILE
)

candidates = pd.read_csv(
    CANDIDATES_FILE
)

evidence = pd.read_csv(
    EVIDENCE_FILE
)

targets = pd.read_csv(
    TARGET_FILE
)

bugfixes = pd.read_csv(
    BUGFIX_FILE
)


# ============================================================
# HIGH-CONFIDENCE CANDIDATES
# ============================================================

candidates["high_confidence"] = (
    candidates["bugfix_title"]
    .astype(str)
    .str.strip()
    .str.upper()
    .str.startswith("BUG")
)

high_candidates = candidates[
    candidates["high_confidence"]
].copy()


# ============================================================
# FUNNEL BY ORIGINAL PR YEAR
# ============================================================

eligible_count = (
    eligible
    .groupby("collection_year")
    .size()
    .rename("eligible_prs")
)


candidate_count = (
    high_candidates[
        [
            "pr_number",
            "collection_year",
        ]
    ]
    .drop_duplicates("pr_number")
    .groupby("collection_year")
    .size()
    .rename("prs_with_candidate")
)


positive_count = (
    targets[
        targets[
            "observed_defect_90d"
        ] == 1
    ]
    .groupby("collection_year")
    .size()
    .rename("positive_prs")
)


funnel = pd.concat(
    [
        eligible_count,
        candidate_count,
        positive_count,
    ],
    axis=1,
).fillna(0)


funnel = funnel.astype(int)


funnel["candidate_rate_pct"] = (
    funnel["prs_with_candidate"]
    / funnel["eligible_prs"]
    * 100
)


funnel["positive_rate_pct"] = (
    funnel["positive_prs"]
    / funnel["eligible_prs"]
    * 100
)


funnel[
    "positive_given_candidate_pct"
] = (
    funnel["positive_prs"]
    / funnel["prs_with_candidate"]
    * 100
)


print()
print("=" * 70)
print("FUNIL TEMPORAL")
print("=" * 70)

print(
    funnel.round(2)
)


# ============================================================
# SZZ EVIDENCE BY YEAR
# ============================================================

evidence_by_year = (
    evidence
    .groupby("collection_year")
    .agg(
        correspondences=(
            "pr_number",
            "size",
        ),

        positive_correspondences=(
            "high_confidence_szz",
            "sum",
        ),
    )
)


evidence_by_year[
    "szz_match_rate_pct"
] = (
    evidence_by_year[
        "positive_correspondences"
    ]
    /
    evidence_by_year[
        "correspondences"
    ]
    * 100
)


print()
print("=" * 70)
print("SZZ POR ANO DA PR ORIGINAL")
print("=" * 70)

print(
    evidence_by_year.round(2)
)


# ============================================================
# BUG FIX INVENTORY
# ============================================================

bugfixes["merged_at"] = pd.to_datetime(
    bugfixes["merged_at"],
    utc=True,
)


bugfixes["bugfix_year"] = (
    bugfixes["merged_at"]
    .dt.year
)


bugfixes["title_starts_bug"] = (
    bugfixes["title"]
    .astype(str)
    .str.strip()
    .str.upper()
    .str.startswith("BUG")
)


bugfix_inventory = (
    bugfixes
    .groupby("bugfix_year")
    .agg(
        bugfix_prs=(
            "pr_number",
            "nunique",
        ),

        title_bug=(
            "title_starts_bug",
            "sum",
        ),
    )
)


print()
print("=" * 70)
print("INVENTÁRIO DE BUG FIXES")
print("=" * 70)

print(
    bugfix_inventory
)


# ============================================================
# MERGE COMMIT STRUCTURE
# ============================================================

print()
print("=" * 70)
print("ESTRUTURA DOS COMMITS DE MERGE")
print("=" * 70)

merge_rows = []


for index, row in eligible.iterrows():

    sha = str(
        row["merge_commit_sha"]
    )

    parents = run_git(
        [
            "show",
            "-s",
            "--format=%P",
            sha,
        ]
    )

    if parents is None:
        parent_count = -1
    elif not parents:
        parent_count = 0
    else:
        parent_count = len(
            parents.split()
        )

    merge_rows.append(
        {
            "pr_number": int(
                row["pr_number"]
            ),

            "collection_year": int(
                row["collection_year"]
            ),

            "parent_count": (
                parent_count
            ),
        }
    )


merge_structure = pd.DataFrame(
    merge_rows
)


merge_summary = (
    merge_structure
    .groupby(
        [
            "collection_year",
            "parent_count",
        ]
    )
    .size()
    .unstack(
        fill_value=0
    )
)


print(
    merge_summary
)


print()
print("=" * 70)
print("AUDITORIA CONCLUÍDA")
print("=" * 70)