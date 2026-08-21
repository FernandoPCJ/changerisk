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
    ROOT / "data" / "raw" / "pandas_pulls_full_enriched.csv"
)

FILES_FILE = (
    ROOT / "data" / "raw" / "pandas_pr_files_full.csv"
)

STATUS_FILE = (
    ROOT / "data" / "raw" / "pandas_pr_files_full_status.csv"
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


# Registra a origem dos dados já extraídos.
if "extraction_source" not in files.columns:

    files["extraction_source"] = "git_local"


represented = set(
    files["pr_number"]
    .astype(int)
)


missing = prs[
    ~prs["pr_number"].isin(
        represented
    )
].copy()


print(
    "PRs ausentes antes do fallback:",
    len(missing),
)

print()


# ============================================================
# GITHUB FALLBACK
# ============================================================

fallback_rows = []


for _, row in missing.iterrows():

    pr_number = int(
        row["pr_number"]
    )

    changed_files = int(
        row["changed_files"]
    )


    # --------------------------------------------------------
    # ZERO FILES
    # --------------------------------------------------------

    if changed_files == 0:

        print(
            f"PR {pr_number}: "
            "changed_files = 0 "
            "→ nenhuma extração necessária"
        )

        continue


    print(
        f"PR {pr_number}: "
        f"{changed_files} arquivos segundo GitHub "
        "→ usando fallback da API"
    )


    page = 1
    collected = 0


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

            wait_seconds = (
                int(retry_after) + 2
                if retry_after
                else 60
            )

            print(
                f"Rate limit. "
                f"Aguardando {wait_seconds}s..."
            )

            time.sleep(
                wait_seconds
            )

            continue


        response.raise_for_status()

        api_files = response.json()


        if not api_files:
            break


        for file in api_files:

            api_status = file["status"]


            status_map = {
                "added": "A",
                "modified": "M",
                "removed": "D",
                "renamed": "R",
                "copied": "C",
                "changed": "M",
                "unchanged": "M",
            }


            filename = file["filename"]

            previous_filename = (
                file.get(
                    "previous_filename"
                )
            )


            fallback_rows.append(
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
                        previous_filename
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
                        "github_api_fallback"
                    ),
                }
            )


        collected += len(
            api_files
        )


        if len(api_files) < 100:
            break


        page += 1


    print(
        f"    → {collected} arquivos coletados"
    )


# ============================================================
# APPEND FALLBACK
# ============================================================

if fallback_rows:

    fallback_df = pd.DataFrame(
        fallback_rows
    )


    files = pd.concat(
        [
            files,
            fallback_df,
        ],
        ignore_index=True,
    )


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


# ============================================================
# STATUS POR PR
# ============================================================

represented_after = set(
    files["pr_number"]
    .astype(int)
)


fallback_prs = set(
    row["pr_number"]
    for row in fallback_rows
)


status_rows = []


for _, row in prs.iterrows():

    pr_number = int(
        row["pr_number"]
    )

    changed_files = int(
        row["changed_files"]
    )


    if pr_number in fallback_prs:

        extraction_status = (
            "success"
        )

        extraction_source = (
            "github_api_fallback"
        )


    elif pr_number in represented_after:

        extraction_status = (
            "success"
        )

        extraction_source = (
            "git_local"
        )


    elif changed_files == 0:

        extraction_status = (
            "success_no_files"
        )

        extraction_source = (
            "not_applicable"
        )


    else:

        extraction_status = (
            "unresolved"
        )

        extraction_source = (
            "unknown"
        )


    status_rows.append(
        {
            "pr_number": (
                pr_number
            ),

            "changed_files_api": (
                changed_files
            ),

            "extraction_status": (
                extraction_status
            ),

            "extraction_source": (
                extraction_source
            ),
        }
    )


status_df = pd.DataFrame(
    status_rows
)


status_df.to_csv(
    STATUS_FILE,
    index=False,
)


# ============================================================
# FINAL VALIDATION
# ============================================================

success = (
    status_df[
        "extraction_status"
    ].isin(
        [
            "success",
            "success_no_files",
        ]
    )
)


unresolved = status_df[
    status_df[
        "extraction_status"
    ] == "unresolved"
]


print()
print("=" * 60)
print("RESOLUÇÃO CONCLUÍDA")
print("=" * 60)

print(
    "PRs totais:",
    len(status_df),
)

print(
    "PRs contabilizadas:",
    int(success.sum()),
)

print(
    "PRs com arquivos:",
    files[
        "pr_number"
    ].nunique(),
)

print(
    "PRs legítimas sem arquivos:",
    int(
        (
            status_df[
                "extraction_status"
            ] == "success_no_files"
        ).sum()
    ),
)

print(
    "PRs resolvidas via fallback:",
    int(
        (
            status_df[
                "extraction_source"
            ] == "github_api_fallback"
        ).sum()
    ),
)

print(
    "PRs não resolvidas:",
    len(unresolved),
)

print(
    "Registros de arquivos:",
    len(files),
)


if not unresolved.empty:

    print()
    print(
        "PRs não resolvidas:"
    )

    print(
        unresolved.to_string(
            index=False
        )
    )


print()
print(
    f"Arquivos atualizados em:\n"
    f"{FILES_FILE}"
)

print()

print(
    f"Status salvo em:\n"
    f"{STATUS_FILE}"
)