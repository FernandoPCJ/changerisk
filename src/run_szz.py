from pathlib import Path
import re
import subprocess

import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

REPO_DIR = ROOT / "repositories" / "pandas"

CANDIDATES_FILE = (
    ROOT / "data" / "processed" / "pandas_defect_candidates_detail.csv"
)

ORIGINAL_PRS_FILE = (
    ROOT / "data" / "raw" / "pandas_pulls_pilot_100_enriched.csv"
)

ORIGINAL_FILES_FILE = (
    ROOT / "data" / "raw" / "pandas_pilot_pr_files.csv"
)

BUGFIX_PRS_FILE = (
    ROOT / "data" / "raw" / "pandas_bugfix_pulls_enriched.csv"
)

EVIDENCE_OUTPUT = (
    ROOT / "data" / "processed" / "pandas_szz_evidence.csv"
)

TARGET_OUTPUT = (
    ROOT / "data" / "processed" / "pandas_pilot_targets.csv"
)


# Último instante coberto pela coleta de bug fixes.
OBSERVATION_CUTOFF = pd.Timestamp(
    "2026-03-31T23:59:59Z"
)


# ============================================================
# GIT HELPERS
# ============================================================

def run_git(arguments):
    """
    Executa um comando Git dentro do repositório local do pandas.
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
    Retorna o primeiro parent de um commit.
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


def get_deleted_ranges(parent_sha, bugfix_sha, filename):
    """
    Identifica intervalos de linhas existentes antes do bug fix
    que foram removidos ou substituídos pela correção.
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

        match = HUNK_PATTERN.match(line)

        if not match:
            continue

        start = int(match.group(1))

        if match.group(2) is None:
            count = 1
        else:
            count = int(match.group(2))

        # count = 0 representa alteração apenas aditiva.
        # Nesse caso, não existe linha anterior para aplicar blame.
        if count == 0:
            continue

        end = start + count - 1

        ranges.append(
            (start, end)
        )

    return ranges


# ============================================================
# BLAME
# ============================================================

def blame_range(parent_sha, filename, start, end):
    """
    Recupera os commits responsáveis pelas linhas existentes
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

        first_token = line.split()[0]

        # Commits de fronteira podem aparecer com ^
        first_token = first_token.lstrip("^")

        if re.fullmatch(
            r"[0-9a-fA-F]{40}",
            first_token,
        ):
            commits.add(
                first_token.lower()
            )

    return commits


# ============================================================
# PRODUCTION FILE
# ============================================================

def is_production_file(filename):
    """
    Considera código de produção do pandas, excluindo testes.
    """

    return (
        filename.startswith("pandas/")
        and not filename.startswith("pandas/tests/")
    )


# ============================================================
# LOAD DATA
# ============================================================

if not REPO_DIR.exists():
    raise RuntimeError(
        f"Repositório do pandas não encontrado em:\n{REPO_DIR}"
    )


candidates = pd.read_csv(
    CANDIDATES_FILE
)

original_prs = pd.read_csv(
    ORIGINAL_PRS_FILE
)

original_files = pd.read_csv(
    ORIGINAL_FILES_FILE
)

bugfix_prs = pd.read_csv(
    BUGFIX_PRS_FILE
)


original_prs["merged_at"] = pd.to_datetime(
    original_prs["merged_at"],
    utc=True,
)

bugfix_prs["merged_at"] = pd.to_datetime(
    bugfix_prs["merged_at"],
    utc=True,
)


# ============================================================
# LOOKUP DICTIONARIES
# ============================================================

original_sha = (
    original_prs
    .set_index("pr_number")["merge_commit_sha"]
    .astype(str)
    .to_dict()
)

bugfix_sha = (
    bugfix_prs
    .set_index("pr_number")["merge_commit_sha"]
    .astype(str)
    .to_dict()
)

bugfix_title = (
    bugfix_prs
    .set_index("pr_number")["title"]
    .astype(str)
    .to_dict()
)


# ============================================================
# RUN SZZ
# ============================================================

evidence_rows = []

total = len(candidates)


for index, row in candidates.iterrows():

    original_pr = int(
        row["pr_number"]
    )

    bugfix_pr = int(
        row["bugfix_pr_number"]
    )

    filename = row["filename"]

    original_commit = (
        original_sha[original_pr]
        .lower()
    )

    bugfix_commit = (
        bugfix_sha[bugfix_pr]
        .lower()
    )

    bugfix_pr_title = (
        bugfix_title[bugfix_pr]
    )

    # Critério adicional de alta confiança:
    # a PR de correção precisa ser explicitamente identificada
    # como BUG no início do título.
    bugfix_title_starts_bug = int(
        bugfix_pr_title
        .strip()
        .upper()
        .startswith("BUG")
    )


    print(
        f"[{index + 1}/{total}] "
        f"PR {original_pr} × Bug {bugfix_pr}"
    )

    print(
        f"    {filename}"
    )


    blamed_commits = set()

    deleted_line_count = 0

    status = "processed"

    error_message = ""


    try:

        parent = get_parent(
            bugfix_commit
        )

        ranges = get_deleted_ranges(
            parent,
            bugfix_commit,
            filename,
        )

        if not ranges:

            status = "no_deleted_lines"

        else:

            for start, end in ranges:

                deleted_line_count += (
                    end - start + 1
                )

                commits = blame_range(
                    parent,
                    filename,
                    start,
                    end,
                )

                blamed_commits.update(
                    commits
                )


    except Exception as exc:

        status = "error"
        error_message = str(exc)


    # Resultado bruto do SZZ.
    szz_match = int(
        original_commit
        in blamed_commits
    )


    if szz_match:

        status = "szz_match"

        print(
            "    → EVIDÊNCIA SZZ POSITIVA"
        )

    else:

        print(
            f"    → sem match ({status})"
        )


    evidence_rows.append(
        {
            "pr_number": original_pr,

            "bugfix_pr_number": bugfix_pr,

            "filename": filename,

            "days_after_merge": row[
                "days_after_merge"
            ],

            "original_commit": (
                original_commit
            ),

            "bugfix_commit": (
                bugfix_commit
            ),

            "bugfix_title": (
                bugfix_pr_title
            ),

            "bugfix_title_starts_bug": (
                bugfix_title_starts_bug
            ),

            "deleted_lines_checked": (
                deleted_line_count
            ),

            "blamed_commits": ";".join(
                sorted(blamed_commits)
            ),

            "szz_match": (
                szz_match
            ),

            "status": (
                status
            ),

            "error": (
                error_message
            ),
        }
    )


# ============================================================
# BUILD EVIDENCE DATAFRAME
# ============================================================

evidence = pd.DataFrame(
    evidence_rows
)


# Evidência de alta confiança:
#
# 1. o SZZ encontrou relação entre a correção e a PR original
# 2. a PR de correção é explicitamente identificada como BUG
#
# Essa regra foi adicionada após a validação manual identificar
# falso positivo causado por uma refatoração que apenas moveu
# código anteriormente introduzido.
evidence["high_confidence_szz"] = (
    (
        evidence["szz_match"] == 1
    )
    &
    (
        evidence[
            "bugfix_title_starts_bug"
        ] == 1
    )
).astype(int)


evidence.to_csv(
    EVIDENCE_OUTPUT,
    index=False,
)


# ============================================================
# BUILD TARGET
# ============================================================

production_prs = set(
    original_files.loc[
        original_files[
            "filename"
        ].apply(
            is_production_file
        ),
        "pr_number",
    ]
)


targets = original_prs[
    [
        "pr_number",
        "title",
        "sample_year",
        "merged_at",
    ]
].copy()


# ------------------------------------------------------------
# PR altera código de produção?
# ------------------------------------------------------------

targets[
    "touches_production_code"
] = (
    targets["pr_number"]
    .isin(production_prs)
    .astype(int)
)


# ------------------------------------------------------------
# Janela de observação
# ------------------------------------------------------------

targets["observation_end"] = (
    targets["merged_at"]
    + pd.Timedelta(days=90)
)


targets[
    "observation_window_complete"
] = (
    targets["observation_end"]
    <= OBSERVATION_CUTOFF
).astype(int)


# ------------------------------------------------------------
# Houve candidato por mesmo arquivo + janela temporal?
# ------------------------------------------------------------

candidate_prs = set(
    candidates["pr_number"]
)


targets[
    "has_candidate_90d"
] = (
    targets["pr_number"]
    .isin(candidate_prs)
    .astype(int)
)


# ============================================================
# RAW SZZ RESULTS
# ============================================================

raw_positive = evidence[
    evidence["szz_match"] == 1
]


raw_positive_bugfixes = (
    raw_positive
    .groupby("pr_number")[
        "bugfix_pr_number"
    ]
    .nunique()
)


raw_positive_files = (
    raw_positive
    .groupby("pr_number")[
        "filename"
    ]
    .nunique()
)


targets[
    "szz_raw_positive_bugfixes"
] = (
    targets["pr_number"]
    .map(raw_positive_bugfixes)
    .fillna(0)
    .astype(int)
)


targets[
    "szz_raw_positive_files"
] = (
    targets["pr_number"]
    .map(raw_positive_files)
    .fillna(0)
    .astype(int)
)


# ============================================================
# HIGH-CONFIDENCE SZZ RESULTS
# ============================================================

positive = evidence[
    evidence[
        "high_confidence_szz"
    ] == 1
]


positive_bugfixes = (
    positive
    .groupby("pr_number")[
        "bugfix_pr_number"
    ]
    .nunique()
)


positive_files = (
    positive
    .groupby("pr_number")[
        "filename"
    ]
    .nunique()
)


targets[
    "szz_positive_bugfixes"
] = (
    targets["pr_number"]
    .map(positive_bugfixes)
    .fillna(0)
    .astype(int)
)


targets[
    "szz_positive_files"
] = (
    targets["pr_number"]
    .map(positive_files)
    .fillna(0)
    .astype(int)
)


# ============================================================
# TARGET ELIGIBILITY
# ============================================================

targets[
    "target_eligible"
] = (
    (
        targets[
            "touches_production_code"
        ] == 1
    )
    &
    (
        targets[
            "observation_window_complete"
        ] == 1
    )
).astype(int)


# ============================================================
# FINAL TARGET
# ============================================================

# O target começa indefinido.
targets[
    "observed_defect_90d"
] = pd.array(
    [pd.NA] * len(targets),
    dtype="Int64",
)


eligible = (
    targets[
        "target_eligible"
    ] == 1
)


# O target final utiliza SOMENTE evidências
# SZZ classificadas como alta confiança.
targets.loc[
    eligible,
    "observed_defect_90d",
] = (
    targets.loc[
        eligible,
        "szz_positive_bugfixes",
    ] > 0
).astype(int)


targets.to_csv(
    TARGET_OUTPUT,
    index=False,
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 60)
print("SZZ CONCLUÍDO")
print("=" * 60)


print(
    "Correspondências de arquivos analisadas:",
    len(evidence),
)


print(
    "Correspondências SZZ brutas:",
    int(
        evidence[
            "szz_match"
        ].sum()
    ),
)


print(
    "Correspondências SZZ de alta confiança:",
    int(
        evidence[
            "high_confidence_szz"
        ].sum()
    ),
)


print(
    "PRs originais com SZZ bruto:",
    raw_positive[
        "pr_number"
    ].nunique(),
)


print(
    "PRs originais com SZZ de alta confiança:",
    positive[
        "pr_number"
    ].nunique(),
)


print(
    "PRs elegíveis para o target:",
    int(
        targets[
            "target_eligible"
        ].sum()
    ),
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
    "Target não aplicável:",
    int(
        targets[
            "observed_defect_90d"
        ].isna().sum()
    ),
)


# ============================================================
# CHECK MANUALLY VALIDATED CASES
# ============================================================

print()
print(
    "Casos previamente verificados manualmente:"
)


manual_cases = [
    # Negativos confirmados
    (50426, 51403),
    (50430, 50085),

    # Positivos confirmados
    (50464, 50242),
    (50464, 50586),
    (50441, 51575),

    # Falso positivo bruto do SZZ:
    # refatoração, não correção de bug.
    (50435, 51525),
]


for original_pr, bugfix_pr in manual_cases:

    result = evidence[
        (
            evidence[
                "pr_number"
            ] == original_pr
        )
        &
        (
            evidence[
                "bugfix_pr_number"
            ] == bugfix_pr
        )
    ]


    if result.empty:

        print(
            f"{original_pr} × {bugfix_pr}: "
            "não encontrado"
        )

    else:

        raw_value = int(
            result[
                "szz_match"
            ].max()
        )

        high_value = int(
            result[
                "high_confidence_szz"
            ].max()
        )

        print(
            f"{original_pr} × {bugfix_pr}: "
            f"SZZ bruto = {raw_value} | "
            f"Alta confiança = {high_value}"
        )


# ============================================================
# OUTPUT FILES
# ============================================================

print()

print(
    f"Evidências salvas em:\n"
    f"{EVIDENCE_OUTPUT}"
)

print()

print(
    f"Targets salvos em:\n"
    f"{TARGET_OUTPUT}"
)