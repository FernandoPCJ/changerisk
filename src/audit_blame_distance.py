from pathlib import Path
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

EVIDENCE_FILE = (
    ROOT
    / "data"
    / "processed"
    / "pandas_szz_full_evidence.csv"
)

OUTPUT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "pandas_blame_distance_audit.csv"
)


# ============================================================
# GIT
# ============================================================

def load_commit_dates():
    """
    Carrega em uma única operação as datas dos commits
    existentes no histórico local do pandas.
    """

    print(
        "Carregando datas dos commits do repositório..."
    )

    result = subprocess.run(
        [
            "git",
            "log",
            "--all",
            "--format=%H|%cI",
        ],
        cwd=REPO_DIR,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    if result.returncode != 0:
        raise RuntimeError(
            result.stderr
        )

    commit_dates = {}

    for line in result.stdout.splitlines():

        if "|" not in line:
            continue

        sha, date = line.split(
            "|",
            1,
        )

        commit_dates[
            sha.lower()
        ] = pd.to_datetime(
            date,
            utc=True,
        )

    print(
        "Commits carregados:",
        len(commit_dates),
    )

    return commit_dates


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(
    EVIDENCE_FILE
)

print(
    "Correspondências SZZ disponíveis:",
    len(df),
)

print()


commit_dates = load_commit_dates()

print()
print(
    "Analisando commits encontrados pelo blame..."
)
print()


# ============================================================
# AUDIT
# ============================================================

rows = []

total = len(df)


for index, row in df.iterrows():

    if (
        (index + 1) % 500 == 0
        or index == 0
        or index + 1 == total
    ):

        print(
            f"[{index + 1}/{total}] "
            "correspondências processadas"
        )


    original_pr = int(
        row["pr_number"]
    )

    year = int(
        row["collection_year"]
    )

    original_commit = str(
        row["original_commit"]
    ).strip().lower()


    blamed_value = row[
        "blamed_commits"
    ]


    if pd.isna(
        blamed_value
    ):
        continue


    original_date = (
        commit_dates.get(
            original_commit
        )
    )


    if original_date is None:
        continue


    blamed_commits = set(
        str(
            blamed_value
        ).split(";")
    )


    for blamed_commit in blamed_commits:

        blamed_commit = (
            blamed_commit
            .strip()
            .lower()
        )


        if not blamed_commit:
            continue


        blamed_date = (
            commit_dates.get(
                blamed_commit
            )
        )


        if blamed_date is None:
            continue


        days_from_original = (
            blamed_date
            - original_date
        ).total_seconds() / 86400


        rows.append(
            {
                "pr_number": (
                    original_pr
                ),

                "collection_year": (
                    year
                ),

                "original_commit": (
                    original_commit
                ),

                "blamed_commit": (
                    blamed_commit
                ),

                "exact_match": int(
                    blamed_commit
                    == original_commit
                ),

                "days_from_original": (
                    days_from_original
                ),
            }
        )


# ============================================================
# DATAFRAME
# ============================================================

result = pd.DataFrame(
    rows
)


if result.empty:

    raise RuntimeError(
        "Nenhuma relação de blame pôde ser analisada."
    )


result.to_csv(
    OUTPUT_FILE,
    index=False,
)


# ============================================================
# EXACT MATCH
# ============================================================

print()
print("=" * 70)
print("DISTÂNCIA ENTRE PR ORIGINAL E BLAME")
print("=" * 70)


summary = (
    result
    .groupby(
        "collection_year"
    )
    .agg(
        blamed_commits=(
            "blamed_commit",
            "count",
        ),

        exact_matches=(
            "exact_match",
            "sum",
        ),

        median_days=(
            "days_from_original",
            "median",
        ),
    )
)


summary[
    "exact_match_pct"
] = (
    summary[
        "exact_matches"
    ]
    /
    summary[
        "blamed_commits"
    ]
    * 100
)


print(
    summary.round(2)
)


# ============================================================
# NEARBY COMMITS
# ============================================================

print()
print("=" * 70)
print("COMMITS POSTERIORES PRÓXIMOS DA PR ORIGINAL")
print("=" * 70)


result[
    "within_30_days"
] = (
    result[
        "days_from_original"
    ]
    .between(
        0,
        30,
        inclusive="both",
    )
)


near = (
    result
    .groupby(
        "collection_year"
    )[
        "within_30_days"
    ]
    .agg(
        [
            "sum",
            "count",
        ]
    )
)


near[
    "pct"
] = (
    near["sum"]
    / near["count"]
    * 100
)


print(
    near.round(2)
)


# ============================================================
# ADDITIONAL WINDOWS
# ============================================================

print()
print("=" * 70)
print("DISTRIBUIÇÃO POR JANELA TEMPORAL")
print("=" * 70)


for days in [
    7,
    14,
    30,
    60,
    90,
]:

    column = (
        f"within_{days}d"
    )

    result[column] = (
        result[
            "days_from_original"
        ]
        .between(
            0,
            days,
            inclusive="both",
        )
    )


    temp = (
        result
        .groupby(
            "collection_year"
        )[column]
        .mean()
        * 100
    )


    print()
    print(
        f"Até {days} dias:"
    )

    print(
        temp.round(2)
    )


print()
print("=" * 70)
print("AUDITORIA CONCLUÍDA")
print("=" * 70)

print()

print(
    "Registros analisados:",
    len(result),
)

print()

print(
    f"Detalhes salvos em:\n"
    f"{OUTPUT_FILE}"
)