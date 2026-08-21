from pathlib import Path
import re
import subprocess

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

REPO_DIR = ROOT / "repositories" / "pandas"

CANDIDATES_FILE = (
    ROOT
    / "data"
    / "processed"
    / "pandas_defect_candidates_full_detail.csv"
)

ELIGIBLE_FILE = (
    ROOT
    / "data"
    / "processed"
    / "pandas_eligible_population.csv"
)

BUGFIX_FILE = (
    ROOT
    / "data"
    / "raw"
    / "pandas_bugfix_pulls_enriched.csv"
)

OUTPUT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "pandas_szz_enhanced_audit.csv"
)


HUNK_PATTERN = re.compile(
    r"^@@ -(\d+)(?:,(\d+))? "
    r"\+\d+(?:,\d+)? @@"
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
        raise RuntimeError(
            f"git {' '.join(args)}\n\n"
            f"{result.stderr}"
        )

    return result.stdout


def get_deleted_ranges(parent, bugfix, filename):

    diff = run_git(
        [
            "diff",
            "--unified=0",
            parent,
            bugfix,
            "--",
            filename,
        ]
    )

    ranges = []

    for line in diff.splitlines():

        match = HUNK_PATTERN.match(line)

        if not match:
            continue

        start = int(match.group(1))

        count = (
            int(match.group(2))
            if match.group(2)
            else 1
        )

        if count == 0:
            continue

        end = start + count - 1

        ranges.append(
            (start, end)
        )

    return ranges


def blame_commits(
    parent,
    filename,
    start,
    end,
    enhanced=False,
):

    args = [
        "blame",
        "-l",
    ]

    if enhanced:
        args.extend(
            [
                "-w",
                "-M",
                "-C",
            ]
        )

    args.extend(
        [
            "-L",
            f"{start},{end}",
            parent,
            "--",
            filename,
        ]
    )

    output = run_git(args)

    commits = set()

    for line in output.splitlines():

        if not line.strip():
            continue

        sha = (
            line.split()[0]
            .lstrip("^")
            .lower()
        )

        if re.fullmatch(
            r"[0-9a-f]{40}",
            sha,
        ):
            commits.add(sha)

    return commits


# ============================================================
# LOAD
# ============================================================

candidates = pd.read_csv(
    CANDIDATES_FILE
)

eligible = pd.read_csv(
    ELIGIBLE_FILE
)

bugfixes = pd.read_csv(
    BUGFIX_FILE
)


# Somente nosso conjunto de alta confiança.
mask = (
    candidates["bugfix_title"]
    .astype(str)
    .str.strip()
    .str.upper()
    .str.startswith("BUG")
)

candidates = candidates[
    mask
].copy()


# Auditoria inicialmente focada em 2024 e 2025.
candidates = candidates[
    candidates["collection_year"]
    .isin([2024, 2025])
].copy()


candidates = (
    candidates
    .drop_duplicates(
        subset=[
            "pr_number",
            "bugfix_pr_number",
            "filename",
        ]
    )
    .reset_index(drop=True)
)


original_sha = (
    eligible
    .set_index("pr_number")[
        "merge_commit_sha"
    ]
    .astype(str)
    .str.lower()
    .to_dict()
)


bugfix_sha = (
    bugfixes
    .set_index("pr_number")[
        "merge_commit_sha"
    ]
    .astype(str)
    .str.lower()
    .to_dict()
)


print()
print("=" * 65)
print("AUDITORIA SZZ APRIMORADO — 2024/2025")
print("=" * 65)

print(
    "Correspondências a analisar:",
    len(candidates),
)

print()


rows = []

total = len(candidates)


for index, row in candidates.iterrows():

    if (
        index == 0
        or (index + 1) % 100 == 0
        or index + 1 == total
    ):
        print(
            f"[{index + 1}/{total}]"
        )


    original_pr = int(
        row["pr_number"]
    )

    bugfix_pr = int(
        row["bugfix_pr_number"]
    )

    filename = str(
        row["filename"]
    )

    original_commit = (
        original_sha[original_pr]
    )

    bugfix_commit = (
        bugfix_sha[bugfix_pr]
    )


    try:

        parent = run_git(
            [
                "rev-parse",
                f"{bugfix_commit}^1",
            ]
        ).strip()


        ranges = get_deleted_ranges(
            parent,
            bugfix_commit,
            filename,
        )


        basic_commits = set()
        enhanced_commits = set()


        for start, end in ranges:

            basic_commits.update(
                blame_commits(
                    parent,
                    filename,
                    start,
                    end,
                    enhanced=False,
                )
            )

            enhanced_commits.update(
                blame_commits(
                    parent,
                    filename,
                    start,
                    end,
                    enhanced=True,
                )
            )


        basic_match = int(
            original_commit
            in basic_commits
        )

        enhanced_match = int(
            original_commit
            in enhanced_commits
        )


        rows.append(
            {
                "pr_number": (
                    original_pr
                ),

                "bugfix_pr_number": (
                    bugfix_pr
                ),

                "collection_year": (
                    int(
                        row[
                            "collection_year"
                        ]
                    )
                ),

                "filename": (
                    filename
                ),

                "basic_match": (
                    basic_match
                ),

                "enhanced_match": (
                    enhanced_match
                ),

                "new_match": int(
                    enhanced_match == 1
                    and basic_match == 0
                ),

                "status": (
                    "processed"
                ),
            }
        )


    except Exception as exc:

        rows.append(
            {
                "pr_number": original_pr,
                "bugfix_pr_number": bugfix_pr,
                "collection_year": int(
                    row["collection_year"]
                ),
                "filename": filename,
                "basic_match": 0,
                "enhanced_match": 0,
                "new_match": 0,
                "status": (
                    f"error: {exc}"
                ),
            }
        )


# ============================================================
# RESULTS
# ============================================================

result = pd.DataFrame(
    rows
)

result.to_csv(
    OUTPUT_FILE,
    index=False,
)


print()
print("=" * 65)
print("RESULTADO")
print("=" * 65)


summary = (
    result
    .groupby("collection_year")
    .agg(
        correspondences=(
            "pr_number",
            "size",
        ),

        basic_matches=(
            "basic_match",
            "sum",
        ),

        enhanced_matches=(
            "enhanced_match",
            "sum",
        ),

        new_matches=(
            "new_match",
            "sum",
        ),
    )
)


print(
    summary
)


print()
print(
    "PRs adicionais encontradas "
    "somente pelo SZZ aprimorado:"
)


new_prs = (
    result[
        result["new_match"] == 1
    ]
    [
        [
            "pr_number",
            "collection_year",
        ]
    ]
    .drop_duplicates()
)


print(
    new_prs[
        "collection_year"
    ]
    .value_counts()
    .sort_index()
)


print()
print(
    "Total de PRs adicionais:",
    new_prs[
        "pr_number"
    ].nunique(),
)


print()
print(
    f"Detalhes salvos em:\n"
    f"{OUTPUT_FILE}"
)