from pathlib import Path
import re
import subprocess

import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

REPO_DIR = (
    ROOT
    / "repositories"
    / "pandas"
)

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

BUGFIX_PRS_FILE = (
    ROOT
    / "data"
    / "raw"
    / "pandas_bugfix_pulls_enriched.csv"
)

CACHE_FILE = (
    ROOT
    / "data"
    / "processed"
    / "pandas_szz_full_blame_cache.csv"
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


CHECKPOINT_EVERY = 25


# ============================================================
# GIT HELPERS
# ============================================================

def run_git(arguments):
    """
    Executa um comando Git no repositório local do pandas.
    """

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
            f"Erro ao executar Git:\n"
            f"git {' '.join(arguments)}\n\n"
            f"{result.stderr}"
        )

    return result.stdout


def get_parent(commit_sha):
    """
    Obtém o primeiro parent de um commit.
    """

    return run_git(
        [
            "rev-parse",
            f"{commit_sha}^1",
        ]
    ).strip()


# ============================================================
# DIFF
# ============================================================

HUNK_PATTERN = re.compile(
    r"^@@ -(\d+)(?:,(\d+))? "
    r"\+\d+(?:,\d+)? @@"
)


def get_deleted_ranges(
    parent_sha,
    bugfix_sha,
    filename,
):
    """
    Retorna os intervalos de linhas existentes antes da
    correção que foram removidos ou substituídos.
    """

    diff = run_git(
        [
            "diff",
            "--unified=0",
            parent_sha,
            bugfix_sha,
            "--",
            filename,
        ]
    )

    ranges = []

    for line in diff.splitlines():

        match = HUNK_PATTERN.match(
            line
        )

        if not match:
            continue

        start = int(
            match.group(1)
        )

        if match.group(2) is None:
            count = 1
        else:
            count = int(
                match.group(2)
            )

        # Alteração puramente aditiva.
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

    return ranges


# ============================================================
# BLAME
# ============================================================

def blame_range(
    parent_sha,
    filename,
    start,
    end,
):
    """
    Obtém os commits responsáveis pelas linhas existentes
    imediatamente antes do bug fix.
    """

    output = run_git(
        [
            "blame",
            "-l",
            "-L",
            f"{start},{end}",
            parent_sha,
            "--",
            filename,
        ]
    )

    commits = set()

    for line in output.splitlines():

        if not line.strip():
            continue

        first_token = (
            line
            .split()[0]
            .lstrip("^")
        )

        if re.fullmatch(
            r"[0-9a-fA-F]{40}",
            first_token,
        ):

            commits.add(
                first_token.lower()
            )

    return commits


# ============================================================
# SAVE CACHE
# ============================================================

def save_cache(rows):
    """
    Salva o cache de blame.
    """

    if not rows:
        return

    cache = pd.DataFrame(
        rows
    )

    cache = (
        cache
        .drop_duplicates(
            subset=[
                "bugfix_pr_number",
                "filename",
            ],
            keep="last",
        )
        .sort_values(
            [
                "bugfix_pr_number",
                "filename",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    cache.to_csv(
        CACHE_FILE,
        index=False,
    )


# ============================================================
# VALIDATION
# ============================================================

if not REPO_DIR.exists():
    raise RuntimeError(
        f"Repositório do pandas não encontrado:\n"
        f"{REPO_DIR}"
    )


# ============================================================
# LOAD
# ============================================================

candidates = pd.read_csv(
    CANDIDATES_FILE
)

eligible = pd.read_csv(
    ELIGIBLE_FILE
)

bugfix_prs = pd.read_csv(
    BUGFIX_PRS_FILE
)


# ============================================================
# HIGH-CONFIDENCE FILTER
# ============================================================

candidates[
    "bugfix_title_starts_bug"
] = (
    candidates[
        "bugfix_title"
    ]
    .astype(str)
    .str.strip()
    .str.upper()
    .str.startswith("BUG")
)


candidates = candidates[
    candidates[
        "bugfix_title_starts_bug"
    ]
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
    .reset_index(
        drop=True
    )
)


print()
print("=" * 60)
print("SZZ DEFINITIVO")
print("=" * 60)

print(
    "Correspondências de alta confiança:",
    len(candidates),
)

print(
    "Pares PR original × bug fix:",
    candidates[
        [
            "pr_number",
            "bugfix_pr_number",
        ]
    ]
    .drop_duplicates()
    .shape[0],
)

print(
    "PRs originais envolvidas:",
    candidates[
        "pr_number"
    ].nunique(),
)


# ============================================================
# LOOKUPS
# ============================================================

original_sha = (
    eligible
    .set_index(
        "pr_number"
    )[
        "merge_commit_sha"
    ]
    .astype(str)
    .str.lower()
    .to_dict()
)


bugfix_sha = (
    bugfix_prs
    .set_index(
        "pr_number"
    )[
        "merge_commit_sha"
    ]
    .astype(str)
    .str.lower()
    .to_dict()
)


# ============================================================
# UNIQUE BUGFIX × FILE GROUPS
# ============================================================

groups = (
    candidates[
        [
            "bugfix_pr_number",
            "filename",
        ]
    ]
    .drop_duplicates()
    .reset_index(
        drop=True
    )
)


print(
    "Combinações únicas bug fix × arquivo:",
    len(groups),
)

print()


# ============================================================
# RESUME CACHE
# ============================================================

if CACHE_FILE.exists():

    existing_cache = pd.read_csv(
        CACHE_FILE
    )

    cache_rows = (
        existing_cache
        .to_dict(
            orient="records"
        )
    )

    completed_keys = set(
        zip(
            existing_cache[
                "bugfix_pr_number"
            ].astype(int),

            existing_cache[
                "filename"
            ].astype(str),
        )
    )

    print(
        "Cache encontrado."
    )

    print(
        "Grupos já processados:",
        len(completed_keys),
    )

else:

    cache_rows = []

    completed_keys = set()

    print(
        "Nenhum cache encontrado."
    )


pending_groups = groups[
    ~groups.apply(
        lambda row: (
            int(
                row[
                    "bugfix_pr_number"
                ]
            ),
            str(
                row[
                    "filename"
                ]
            ),
        )
        in completed_keys,
        axis=1,
    )
].copy()


print(
    "Grupos pendentes:",
    len(pending_groups),
)

print()


# ============================================================
# BUILD BLAME CACHE
# ============================================================

processed_since_checkpoint = 0

already_done = len(
    completed_keys
)

total_groups = len(
    groups
)


for position, (_, row) in enumerate(
    pending_groups.iterrows(),
    start=1,
):

    bugfix_pr = int(
        row[
            "bugfix_pr_number"
        ]
    )

    filename = str(
        row[
            "filename"
        ]
    )

    absolute_position = (
        already_done
        + position
    )


    print(
        f"[{absolute_position}/{total_groups}] "
        f"Bug {bugfix_pr}"
    )

    print(
        f"    {filename}"
    )


    blamed_commits = set()

    deleted_lines_checked = 0

    status = "processed"

    error_message = ""

    parent_commit = ""


    try:

        bugfix_commit = (
            bugfix_sha[
                bugfix_pr
            ]
        )

        parent_commit = (
            get_parent(
                bugfix_commit
            )
        )


        ranges = (
            get_deleted_ranges(
                parent_commit,
                bugfix_commit,
                filename,
            )
        )


        if not ranges:

            status = (
                "no_deleted_lines"
            )


        else:

            for start, end in ranges:

                deleted_lines_checked += (
                    end
                    - start
                    + 1
                )

                commits = (
                    blame_range(
                        parent_commit,
                        filename,
                        start,
                        end,
                    )
                )

                blamed_commits.update(
                    commits
                )


    except Exception as exc:

        status = "error"

        error_message = str(
            exc
        )


    cache_rows.append(
        {
            "bugfix_pr_number": (
                bugfix_pr
            ),

            "filename": (
                filename
            ),

            "bugfix_commit": (
                bugfix_sha.get(
                    bugfix_pr,
                    "",
                )
            ),

            "parent_commit": (
                parent_commit
            ),

            "deleted_lines_checked": (
                deleted_lines_checked
            ),

            "blamed_commits": (
                ";".join(
                    sorted(
                        blamed_commits
                    )
                )
            ),

            "status": (
                status
            ),

            "error": (
                error_message
            ),
        }
    )


    if status == "error":

        print(
            "    → ERRO"
        )

    elif status == "no_deleted_lines":

        print(
            "    → nenhuma linha removida"
        )

    else:

        print(
            "    → blame concluído"
        )


    processed_since_checkpoint += 1


    # ========================================================
    # CHECKPOINT
    # ========================================================

    if (
        processed_since_checkpoint
        >= CHECKPOINT_EVERY
    ):

        save_cache(
            cache_rows
        )

        print()
        print(
            "CHECKPOINT SALVO"
        )
        print()

        processed_since_checkpoint = 0


# ============================================================
# FINAL CACHE SAVE
# ============================================================

save_cache(
    cache_rows
)


cache = pd.read_csv(
    CACHE_FILE
)


# ============================================================
# CREATE EVIDENCE
# ============================================================

evidence = candidates.merge(
    cache[
        [
            "bugfix_pr_number",
            "filename",
            "bugfix_commit",
            "parent_commit",
            "deleted_lines_checked",
            "blamed_commits",
            "status",
            "error",
        ]
    ],
    on=[
        "bugfix_pr_number",
        "filename",
    ],
    how="left",
)


evidence[
    "original_commit"
] = (
    evidence[
        "pr_number"
    ]
    .map(
        original_sha
    )
)


def has_szz_match(row):
    """
    Verifica se o commit da PR original aparece
    entre os commits retornados pelo blame.
    """

    original_commit = str(
        row[
            "original_commit"
        ]
    ).lower()

    blamed_value = row[
        "blamed_commits"
    ]

    if pd.isna(
        blamed_value
    ):
        return 0

    blamed = set(
        str(
            blamed_value
        ).split(";")
    )

    return int(
        original_commit
        in blamed
    )


evidence[
    "high_confidence_szz"
] = evidence.apply(
    has_szz_match,
    axis=1,
)


evidence.to_csv(
    EVIDENCE_FILE,
    index=False,
)


# ============================================================
# BUILD FINAL TARGET
# ============================================================

targets = eligible.copy()


candidate_prs = set(
    candidates[
        "pr_number"
    ].astype(int)
)


targets[
    "has_high_confidence_candidate_90d"
] = (
    targets[
        "pr_number"
    ]
    .isin(
        candidate_prs
    )
    .astype(int)
)


# ============================================================
# POSITIVE EVIDENCE
# ============================================================

positive = evidence[
    evidence[
        "high_confidence_szz"
    ] == 1
].copy()


positive_bugfixes = (
    positive
    .groupby(
        "pr_number"
    )[
        "bugfix_pr_number"
    ]
    .nunique()
)


positive_files = (
    positive
    .groupby(
        "pr_number"
    )[
        "filename"
    ]
    .nunique()
)


targets[
    "szz_positive_bugfixes"
] = (
    targets[
        "pr_number"
    ]
    .map(
        positive_bugfixes
    )
    .fillna(0)
    .astype(int)
)


targets[
    "szz_positive_files"
] = (
    targets[
        "pr_number"
    ]
    .map(
        positive_files
    )
    .fillna(0)
    .astype(int)
)


# ============================================================
# UNRESOLVED ERRORS
# ============================================================

error_prs = set(
    evidence.loc[
        evidence[
            "status"
        ] == "error",
        "pr_number",
    ].astype(int)
)


targets[
    "szz_processing_complete"
] = (
    ~targets[
        "pr_number"
    ]
    .isin(
        error_prs
    )
).astype(int)


# ============================================================
# FINAL TARGET
# ============================================================

targets[
    "observed_defect_90d"
] = pd.array(
    [pd.NA] * len(targets),
    dtype="Int64",
)


ready = (
    targets[
        "szz_processing_complete"
    ] == 1
)


targets.loc[
    ready,
    "observed_defect_90d",
] = (
    targets.loc[
        ready,
        "szz_positive_bugfixes",
    ] > 0
).astype(int)


targets.to_csv(
    TARGET_FILE,
    index=False,
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 60)
print("SZZ DEFINITIVO CONCLUÍDO")
print("=" * 60)


print(
    "PRs elegíveis:",
    len(targets),
)


print(
    "Correspondências de alta confiança analisadas:",
    len(evidence),
)


print(
    "Combinações únicas bug fix × arquivo:",
    len(cache),
)


print(
    "Erros no processamento:",
    int(
        (
            cache[
                "status"
            ] == "error"
        ).sum()
    ),
)


print(
    "Correspondências com SZZ positivo:",
    int(
        evidence[
            "high_confidence_szz"
        ].sum()
    ),
)


print(
    "PRs com SZZ positivo:",
    positive[
        "pr_number"
    ].nunique(),
)


print(
    "observed_defect_90d = 1:",
    int(
        (
            targets[
                "observed_defect_90d"
            ] == 1
        ).sum()
    ),
)


print(
    "observed_defect_90d = 0:",
    int(
        (
            targets[
                "observed_defect_90d"
            ] == 0
        ).sum()
    ),
)


print(
    "Target indefinido por erro:",
    int(
        targets[
            "observed_defect_90d"
        ].isna()
        .sum()
    ),
)


print()
print(
    "Distribuição do target por ano:"
)


valid_targets = targets[
    targets[
        "observed_defect_90d"
    ].notna()
].copy()


distribution = (
    valid_targets
    .groupby(
        [
            "collection_year",
            "observed_defect_90d",
        ]
    )
    .size()
    .unstack(
        fill_value=0
    )
)


print(
    distribution
)


print()

print(
    f"Cache:\n"
    f"{CACHE_FILE}"
)

print()

print(
    f"Evidências:\n"
    f"{EVIDENCE_FILE}"
)

print()

print(
    f"Target final:\n"
    f"{TARGET_FILE}"
)