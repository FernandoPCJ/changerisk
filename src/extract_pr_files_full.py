from pathlib import Path
import subprocess

import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

REPO_DIR = ROOT / "repositories" / "pandas"

INPUT_FILE = (
    ROOT / "data" / "raw" / "pandas_pulls_full_enriched.csv"
)

OUTPUT_FILE = (
    ROOT / "data" / "raw" / "pandas_pr_files_full.csv"
)

ERROR_FILE = (
    ROOT / "data" / "raw" / "pandas_pr_files_full_errors.csv"
)

CHECKPOINT_EVERY = 100


# ============================================================
# GIT
# ============================================================

def run_git(arguments):
    result = subprocess.run(
        ["git", *arguments],
        cwd=REPO_DIR,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(arguments)}\n\n"
            f"{result.stderr}"
        )

    return result.stdout


def get_parent(commit_sha):
    return run_git(
        [
            "rev-parse",
            f"{commit_sha}^1",
        ]
    ).strip()


def get_changed_files(parent_sha, commit_sha):
    """
    Retorna os arquivos modificados pelo commit
    em relação ao primeiro parent.
    """

    output = run_git(
        [
            "diff",
            "--name-status",
            "-M",
            parent_sha,
            commit_sha,
            "--",
        ]
    )

    files = []

    for line in output.splitlines():

        if not line.strip():
            continue

        parts = line.split("\t")

        status = parts[0]

        # Rename:
        # R100    old_path    new_path
        if status.startswith("R") and len(parts) >= 3:

            previous_filename = parts[1]
            filename = parts[2]

        else:

            previous_filename = None
            filename = parts[-1]

        files.append(
            {
                "filename": filename,
                "status": status,
                "previous_filename": previous_filename,
            }
        )

    return files


# ============================================================
# SAVE
# ============================================================

def save_checkpoint(rows):

    if not rows:
        return

    df = pd.DataFrame(rows)

    df = (
        df
        .drop_duplicates(
            subset=[
                "pr_number",
                "filename",
            ],
            keep="last",
        )
        .sort_values(
            [
                "pr_number",
                "filename",
            ]
        )
        .reset_index(drop=True)
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )


def save_errors(errors):

    if not errors:
        return

    pd.DataFrame(
        errors
    ).to_csv(
        ERROR_FILE,
        index=False,
    )


# ============================================================
# VALIDATION
# ============================================================

if not REPO_DIR.exists():
    raise RuntimeError(
        f"Repositório não encontrado:\n{REPO_DIR}"
    )


prs = pd.read_csv(
    INPUT_FILE
)


print(
    "PRs disponíveis:",
    len(prs),
)


# ============================================================
# RESUME
# ============================================================

if OUTPUT_FILE.exists():

    existing = pd.read_csv(
        OUTPUT_FILE
    )

    completed_prs = set(
        existing["pr_number"]
        .astype(int)
    )

    rows = existing.to_dict(
        orient="records"
    )

    print(
        "Checkpoint encontrado."
    )

    print(
        "PRs já processadas:",
        len(completed_prs),
    )

else:

    completed_prs = set()
    rows = []

    print(
        "Nenhum checkpoint encontrado."
    )


pending = prs[
    ~prs["pr_number"].isin(
        completed_prs
    )
].copy()


print(
    "PRs pendentes:",
    len(pending),
)

print()


# ============================================================
# EXTRACTION
# ============================================================

errors = []

processed_since_checkpoint = 0

total = len(prs)

already_done = len(
    completed_prs
)


for position, (_, row) in enumerate(
    pending.iterrows(),
    start=1,
):

    pr_number = int(
        row["pr_number"]
    )

    commit_sha = str(
        row["merge_commit_sha"]
    )

    absolute_position = (
        already_done + position
    )

    try:

        parent_sha = get_parent(
            commit_sha
        )

        files = get_changed_files(
            parent_sha,
            commit_sha,
        )

        for file in files:

            rows.append(
                {
                    "pr_number": pr_number,

                    "merge_commit_sha": (
                        commit_sha
                    ),

                    "collection_year": (
                        row["collection_year"]
                    ),

                    "filename": (
                        file["filename"]
                    ),

                    "status": (
                        file["status"]
                    ),

                    "previous_filename": (
                        file[
                            "previous_filename"
                        ]
                    ),
                }
            )

        print(
            f"[{absolute_position}/{total}] "
            f"PR {pr_number}: "
            f"{len(files)} arquivos"
        )

        processed_since_checkpoint += 1


    except Exception as exc:

        print(
            f"[ERRO] PR {pr_number}: "
            f"{exc}"
        )

        errors.append(
            {
                "pr_number": pr_number,
                "merge_commit_sha": commit_sha,
                "error": str(exc),
            }
        )


    if (
        processed_since_checkpoint
        >= CHECKPOINT_EVERY
    ):

        save_checkpoint(
            rows
        )

        save_errors(
            errors
        )

        print()
        print(
            "CHECKPOINT SALVO"
        )
        print()

        processed_since_checkpoint = 0


# ============================================================
# FINAL SAVE
# ============================================================

save_checkpoint(
    rows
)

save_errors(
    errors
)


files_df = pd.read_csv(
    OUTPUT_FILE
)


# ============================================================
# CLASSIFICATION
# ============================================================

files_df[
    "is_production_code"
] = (
    files_df["filename"]
    .str.startswith("pandas/")
    &
    ~files_df["filename"]
    .str.startswith("pandas/tests/")
)


files_df[
    "is_test"
] = (
    files_df["filename"]
    .str.startswith("pandas/tests/")
)


files_df[
    "is_documentation"
] = (
    files_df["filename"]
    .str.startswith("doc/")
)


files_df.to_csv(
    OUTPUT_FILE,
    index=False,
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 60)
print("EXTRAÇÃO CONCLUÍDA")
print("=" * 60)

print(
    "PRs na população:",
    total,
)

print(
    "PRs representadas:",
    files_df[
        "pr_number"
    ].nunique(),
)

print(
    "Registros de arquivos:",
    len(files_df),
)

print(
    "Arquivos de código de produção:",
    int(
        files_df[
            "is_production_code"
        ].sum()
    ),
)

print(
    "Arquivos de teste:",
    int(
        files_df[
            "is_test"
        ].sum()
    ),
)

print(
    "Arquivos de documentação:",
    int(
        files_df[
            "is_documentation"
        ].sum()
    ),
)

print(
    "Erros:",
    len(errors),
)

print()

print(
    f"Arquivo salvo em:\n"
    f"{OUTPUT_FILE}"
)