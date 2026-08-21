from pathlib import Path
import os
import time

import pandas as pd
import requests
from dotenv import load_dotenv


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

PRS_FILE = (
    ROOT
    / "data"
    / "raw"
    / "pandas_pulls_full_enriched.csv"
)

FILES_FILE = (
    ROOT
    / "data"
    / "raw"
    / "pandas_pr_files_full.csv"
)

AUDIT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "pandas_pr_files_reconciliation.csv"
)


OWNER = "pandas-dev"
REPO = "pandas"


# ============================================================
# AUTH
# ============================================================

load_dotenv()

token = os.getenv("GITHUB_TOKEN")

if not token:
    raise RuntimeError(
        "GITHUB_TOKEN não encontrado no arquivo .env"
    )


session = requests.Session()

session.headers.update(
    {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
    }
)


# ============================================================
# LOAD
# ============================================================

prs = pd.read_csv(
    PRS_FILE
)

files = pd.read_csv(
    FILES_FILE
)


# ============================================================
# IDENTIFICAR DIVERGÊNCIAS
# ============================================================

local_counts = (
    files
    .groupby("pr_number")["filename"]
    .nunique()
    .rename("extracted_files")
)


check = prs.merge(
    local_counts,
    on="pr_number",
    how="left",
)


check["extracted_files"] = (
    check["extracted_files"]
    .fillna(0)
    .astype(int)
)


check["changed_files"] = (
    pd.to_numeric(
        check["changed_files"],
        errors="coerce",
    )
    .fillna(0)
    .astype(int)
)


divergent = check[
    check["changed_files"]
    != check["extracted_files"]
].copy()


print()
print("=" * 70)
print("RECONCILIAÇÃO DOS ARQUIVOS DAS PRs")
print("=" * 70)

print(
    "PRs totais:",
    len(check),
)

print(
    "Divergências encontradas:",
    len(divergent),
)

print()


# ============================================================
# GITHUB FILES
# ============================================================

def get_pr_files(pr_number):

    rows = []
    page = 1

    while True:

        url = (
            f"https://api.github.com/repos/"
            f"{OWNER}/{REPO}/pulls/"
            f"{pr_number}/files"
        )

        response = session.get(
            url,
            params={
                "per_page": 100,
                "page": page,
            },
            timeout=30,
        )


        if response.status_code in (
            403,
            429,
        ):

            retry_after = (
                response.headers.get(
                    "Retry-After"
                )
            )

            if retry_after:

                wait_seconds = (
                    int(retry_after) + 2
                )

            else:

                reset = (
                    response.headers.get(
                        "X-RateLimit-Reset"
                    )
                )

                if reset:

                    wait_seconds = max(
                        int(reset)
                        - int(time.time())
                        + 5,
                        10,
                    )

                else:

                    wait_seconds = 60


            print(
                f"Rate limit. "
                f"Aguardando {wait_seconds}s..."
            )

            time.sleep(
                wait_seconds
            )

            continue


        response.raise_for_status()

        data = response.json()


        if not data:
            break


        rows.extend(
            data
        )


        if len(data) < 100:
            break


        page += 1


    return rows


# ============================================================
# RECONCILIAR
# ============================================================

audit_rows = []


for position, (_, row) in enumerate(
    divergent.iterrows(),
    start=1,
):

    pr_number = int(
        row["pr_number"]
    )

    expected = int(
        row["changed_files"]
    )

    before = int(
        row["extracted_files"]
    )


    print(
        f"[{position}/{len(divergent)}] "
        f"PR {pr_number}"
    )

    print(
        f"    API resumida: {expected}"
    )

    print(
        f"    Git local:     {before}"
    )


    api_files = get_pr_files(
        pr_number
    )


    api_count = len(
        api_files
    )


    print(
        f"    API /files:    {api_count}"
    )


    # Remove registros antigos dessa PR.
    files = files[
        files["pr_number"]
        != pr_number
    ].copy()


    new_rows = []


    for item in api_files:

        filename = item[
            "filename"
        ]

        api_status = item[
            "status"
        ]


        status_map = {
            "added": "A",
            "modified": "M",
            "removed": "D",
            "renamed": "R",
            "copied": "C",
            "changed": "M",
            "unchanged": "M",
        }


        new_rows.append(
            {
                "pr_number": (
                    pr_number
                ),

                "merge_commit_sha": (
                    row[
                        "merge_commit_sha"
                    ]
                ),

                "collection_year": (
                    row[
                        "collection_year"
                    ]
                ),

                "filename": (
                    filename
                ),

                "status": (
                    status_map.get(
                        api_status,
                        api_status,
                    )
                ),

                "previous_filename": (
                    item.get(
                        "previous_filename"
                    )
                ),

                "is_production_code": (
                    filename.startswith(
                        "pandas/"
                    )
                    and not filename.startswith(
                        "pandas/tests/"
                    )
                ),

                "is_test": (
                    filename.startswith(
                        "pandas/tests/"
                    )
                ),

                "is_documentation": (
                    filename.startswith(
                        "doc/"
                    )
                ),

                "extraction_source": (
                    "github_api_reconciliation"
                ),
            }
        )


    if new_rows:

        files = pd.concat(
            [
                files,
                pd.DataFrame(
                    new_rows
                ),
            ],
            ignore_index=True,
        )


    audit_rows.append(
        {
            "pr_number": (
                pr_number
            ),

            "changed_files_summary_api": (
                expected
            ),

            "files_before_reconciliation": (
                before
            ),

            "files_from_files_api": (
                api_count
            ),

            "resolved": (
                api_count == expected
            ),
        }
    )


# ============================================================
# SAVE FILES
# ============================================================

files = (
    files
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
    .reset_index(
        drop=True
    )
)


files.to_csv(
    FILES_FILE,
    index=False,
)


audit = pd.DataFrame(
    audit_rows
)


audit.to_csv(
    AUDIT_FILE,
    index=False,
)


# ============================================================
# FINAL VALIDATION
# ============================================================

new_counts = (
    files
    .groupby("pr_number")[
        "filename"
    ]
    .nunique()
    .rename(
        "extracted_files"
    )
)


validation = prs.merge(
    new_counts,
    on="pr_number",
    how="left",
)


validation[
    "extracted_files"
] = (
    validation[
        "extracted_files"
    ]
    .fillna(0)
    .astype(int)
)


validation[
    "changed_files"
] = (
    pd.to_numeric(
        validation[
            "changed_files"
        ],
        errors="coerce",
    )
    .fillna(0)
    .astype(int)
)


remaining = validation[
    validation[
        "changed_files"
    ]
    != validation[
        "extracted_files"
    ]
]


print()
print("=" * 70)
print("RECONCILIAÇÃO CONCLUÍDA")
print("=" * 70)

print(
    "PRs reconciliadas:",
    len(audit),
)

print(
    "Reconciliações que bateram com a API:",
    int(
        audit[
            "resolved"
        ].sum()
    ),
)

print(
    "Divergências restantes:",
    len(remaining),
)

print(
    "Registros finais de arquivos:",
    len(files),
)


if not remaining.empty:

    print()
    print(
        "Divergências ainda existentes:"
    )

    print(
        remaining[
            [
                "pr_number",
                "collection_year",
                "changed_files",
                "extracted_files",
            ]
        ].to_string(
            index=False
        )
    )


print()
print(
    f"Auditoria salva em:\n"
    f"{AUDIT_FILE}"
)