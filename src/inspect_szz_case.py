from pathlib import Path
import re
import subprocess

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPO_DIR = ROOT / "repositories" / "pandas"

ORIGINAL_PR = 50435
BUGFIX_PR = 51525

FILE = "pandas/_libs/tslibs/np_datetime.pxd"

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


# ==========================================
# CARREGAR OS SHAs DIRETAMENTE DOS CSVs
# ==========================================

original_df = pd.read_csv(
    ROOT / "data/raw/pandas_pulls_pilot_100_enriched.csv"
)

bugfix_df = pd.read_csv(
    ROOT / "data/raw/pandas_bugfix_pulls_enriched.csv"
)


ORIGINAL = original_df.loc[
    original_df["pr_number"] == ORIGINAL_PR,
    "merge_commit_sha",
].iloc[0]

BUGFIX = bugfix_df.loc[
    bugfix_df["pr_number"] == BUGFIX_PR,
    "merge_commit_sha",
].iloc[0]


print("PR original:", ORIGINAL_PR)
print("Commit original:", ORIGINAL)

print()

print("Bug fix:", BUGFIX_PR)
print("Commit bug fix:", BUGFIX)

print()


# ==========================================
# VALIDAR COMMITS
# ==========================================

print(
    "Tipo original:",
    run_git(["cat-file", "-t", ORIGINAL]).strip(),
)

print(
    "Tipo bugfix:",
    run_git(["cat-file", "-t", BUGFIX]).strip(),
)

print()


parent = run_git(
    ["rev-parse", f"{BUGFIX}^1"]
).strip()


print("Parent do bugfix:", parent)
print()


# ==========================================
# DIFF
# ==========================================

diff = run_git(
    [
        "diff",
        "--unified=0",
        parent,
        BUGFIX,
        "--",
        FILE,
    ]
)


pattern = re.compile(
    r"^@@ -(\d+)(?:,(\d+))? "
    r"\+\d+(?:,\d+)? @@"
)


matches = []


for line in diff.splitlines():

    match = pattern.match(line)

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

    blame = run_git(
        [
            "blame",
            "-l",
            "-L",
            f"{start},{end}",
            parent,
            "--",
            FILE,
        ]
    )

    for blame_line in blame.splitlines():

        sha = blame_line.split()[0].lstrip("^")

        if sha.lower() == ORIGINAL.lower():

            matches.append(
                {
                    "start": start,
                    "end": end,
                    "blame": blame_line,
                }
            )


print("=" * 60)
print("RESULTADO")
print("=" * 60)

print(
    "Evidências encontradas:",
    len(matches)
)

print()

for item in matches:

    print(
        f"Intervalo removido: "
        f"{item['start']}–{item['end']}"
    )

    print(item["blame"])
    print()