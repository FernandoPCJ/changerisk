from pathlib import Path
import argparse
import re
import subprocess

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPO_DIR = ROOT / "repositories" / "pandas"

ORIGINAL_FILE = (
    ROOT / "data/raw/pandas_pulls_full_enriched.csv"
)

BUGFIX_FILE = (
    ROOT / "data/raw/pandas_bugfix_pulls_enriched.csv"
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
            f"git {' '.join(args)}\n\n{result.stderr}"
        )

    return result.stdout


parser = argparse.ArgumentParser()

parser.add_argument(
    "original_pr",
    type=int,
)

parser.add_argument(
    "bugfix_pr",
    type=int,
)

parser.add_argument(
    "filename",
)

args = parser.parse_args()


original_df = pd.read_csv(
    ORIGINAL_FILE
)

bugfix_df = pd.read_csv(
    BUGFIX_FILE
)


original_row = original_df[
    original_df["pr_number"]
    == args.original_pr
].iloc[0]


bugfix_row = bugfix_df[
    bugfix_df["pr_number"]
    == args.bugfix_pr
].iloc[0]


original_commit = str(
    original_row["merge_commit_sha"]
).lower()

bugfix_commit = str(
    bugfix_row["merge_commit_sha"]
).lower()


print()
print("=" * 70)
print("INSPEÇÃO SZZ")
print("=" * 70)

print(
    "PR original:",
    args.original_pr,
)

print(
    "Título:",
    original_row["title"],
)

print(
    "Commit:",
    original_commit,
)

print()

print(
    "Bug fix:",
    args.bugfix_pr,
)

print(
    "Título:",
    bugfix_row["title"],
)

print(
    "Commit:",
    bugfix_commit,
)

print()

print(
    "Arquivo:",
    args.filename,
)


# ============================================================
# BUGFIX
# ============================================================

bugfix_parent = run_git(
    [
        "rev-parse",
        f"{bugfix_commit}^1",
    ]
).strip()


diff = run_git(
    [
        "diff",
        "--unified=0",
        bugfix_parent,
        bugfix_commit,
        "--",
        args.filename,
    ]
)


pattern = re.compile(
    r"^@@ -(\d+)(?:,(\d+))? "
    r"\+\d+(?:,\d+)? @@"
)


ranges = []


for line in diff.splitlines():

    match = pattern.match(line)

    if not match:
        continue

    start = int(
        match.group(1)
    )

    count = (
        int(match.group(2))
        if match.group(2)
        else 1
    )

    if count == 0:
        continue

    end = (
        start
        + count
        - 1
    )

    ranges.append(
        (start, end)
    )


print()
print("=" * 70)
print("LINHAS REMOVIDAS PELO BUG FIX")
print("=" * 70)


if not ranges:

    print(
        "Nenhuma linha anterior foi removida."
    )


original_found = False


for start, end in ranges:

    print()
    print(
        f"Intervalo: {start}-{end}"
    )

    blame = run_git(
        [
            "blame",
            "-l",
            "-L",
            f"{start},{end}",
            bugfix_parent,
            "--",
            args.filename,
        ]
    )


    for line in blame.splitlines():

        sha = (
            line.split()[0]
            .lstrip("^")
            .lower()
        )

        marker = ""

        if sha == original_commit:

            marker = (
                "  <<< PR ORIGINAL"
            )

            original_found = True

        print(
            line + marker
        )


# ============================================================
# BUGFIX DIFF
# ============================================================

print()
print("=" * 70)
print("DIFF DO BUG FIX")
print("=" * 70)

bugfix_diff = run_git(
    [
        "diff",
        "--unified=3",
        bugfix_parent,
        bugfix_commit,
        "--",
        args.filename,
    ]
)

print(
    bugfix_diff
)


# ============================================================
# ORIGINAL DIFF
# ============================================================

original_parent = run_git(
    [
        "rev-parse",
        f"{original_commit}^1",
    ]
).strip()


print()
print("=" * 70)
print("DIFF DA PR ORIGINAL")
print("=" * 70)

original_diff = run_git(
    [
        "diff",
        "--unified=3",
        original_parent,
        original_commit,
        "--",
        args.filename,
    ]
)

print(
    original_diff
)


# ============================================================
# RESULT
# ============================================================

print()
print("=" * 70)
print("RESULTADO AUTOMÁTICO")
print("=" * 70)

print(
    "Commit original encontrado no blame:",
    original_found,
)