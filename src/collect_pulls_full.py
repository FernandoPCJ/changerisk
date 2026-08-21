import os
import time
from calendar import monthrange

import pandas as pd
import requests
from dotenv import load_dotenv


OWNER = "pandas-dev"
REPO = "pandas"

START_YEAR = 2022
END_YEAR = 2025

OUTPUT_FILE = "data/raw/pandas_pulls_full.csv"

SEARCH_URL = "https://api.github.com/search/issues"


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
# COLETA
# ============================================================

rows = []


for year in range(
    START_YEAR,
    END_YEAR + 1,
):

    for month in range(1, 13):

        last_day = monthrange(
            year,
            month,
        )[1]

        start_date = (
            f"{year}-{month:02d}-01"
        )

        end_date = (
            f"{year}-{month:02d}-{last_day:02d}"
        )

        print()
        print(
            f"Coletando "
            f"{start_date} → {end_date}"
        )

        page = 1
        month_rows = []


        while True:

            params = {
                "q": (
                    f"repo:{OWNER}/{REPO} "
                    "is:pr "
                    "is:merged "
                    f"merged:{start_date}..{end_date}"
                ),
                "sort": "created",
                "order": "asc",
                "per_page": 100,
                "page": page,
            }


            response = session.get(
                SEARCH_URL,
                params=params,
                timeout=30,
            )


            # ------------------------------------------------
            # Rate limit
            # ------------------------------------------------

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
                        int(retry_after) + 1
                    )

                else:

                    wait_seconds = 65

                print(
                    f"Rate limit atingido. "
                    f"Aguardando {wait_seconds}s..."
                )

                time.sleep(
                    wait_seconds
                )

                continue


            response.raise_for_status()

            data = response.json()

            total_count = data[
                "total_count"
            ]

            items = data[
                "items"
            ]


            if page == 1:

                print(
                    "PRs encontradas no mês:",
                    total_count,
                )

                if total_count > 1000:

                    raise RuntimeError(
                        f"O período "
                        f"{start_date} → "
                        f"{end_date} possui "
                        f"{total_count} resultados. "
                        "Divida o período em "
                        "intervalos menores."
                    )


            if not items:
                break


            for item in items:

                month_rows.append(
                    {
                        "pr_number": (
                            item["number"]
                        ),

                        "title": (
                            item["title"]
                        ),

                        "author": (
                            item["user"]["login"]
                        ),

                        "created_at": (
                            item["created_at"]
                        ),

                        "closed_at": (
                            item["closed_at"]
                        ),

                        "comments": (
                            item["comments"]
                        ),

                        "labels": [
                            label["name"]
                            for label
                            in item["labels"]
                        ],

                        "collection_year": (
                            year
                        ),

                        "collection_month": (
                            month
                        ),
                    }
                )


            print(
                f"Página {page}: "
                f"{len(items)} PRs"
            )


            if len(items) < 100:
                break


            page += 1

            # Search API possui limites próprios.
            # Pequeno intervalo reduz risco de
            # rate limit secundário.
            time.sleep(2.2)


        rows.extend(
            month_rows
        )


        # ----------------------------------------------------
        # CHECKPOINT
        # ----------------------------------------------------

        checkpoint = pd.DataFrame(
            rows
        )

        checkpoint = (
            checkpoint
            .drop_duplicates(
                subset="pr_number"
            )
        )

        checkpoint.to_csv(
            OUTPUT_FILE,
            index=False,
        )


        print(
            f"Total acumulado: "
            f"{len(checkpoint)}"
        )

        # Evita excesso de chamadas consecutivas.
        time.sleep(2.2)


# ============================================================
# FINAL
# ============================================================

df = pd.DataFrame(
    rows
)

df = (
    df
    .drop_duplicates(
        subset="pr_number"
    )
    .sort_values(
        "pr_number"
    )
    .reset_index(
        drop=True
    )
)


df.to_csv(
    OUTPUT_FILE,
    index=False,
)


print()
print("=" * 60)
print("COLETA COMPLETA")
print("=" * 60)

print(
    "PRs únicas coletadas:",
    len(df),
)

print()

print(
    "PRs por ano:"
)

print(
    df[
        "collection_year"
    ]
    .value_counts()
    .sort_index()
)

print()

print(
    f"Arquivo salvo em:\n"
    f"{OUTPUT_FILE}"
)