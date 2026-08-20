import os
import time

import pandas as pd
import requests
from dotenv import load_dotenv


OWNER = "pandas-dev"
REPO = "pandas"

OUTPUT_FILE = "data/raw/pandas_bugfix_pulls.csv"

DATE_RANGES = [
    ("2022-01-01", "2022-12-31"),
    ("2023-01-01", "2023-12-31"),
    ("2024-01-01", "2024-12-31"),
    ("2025-01-01", "2025-12-31"),
    ("2026-01-01", "2026-03-31"),
]


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


SEARCH_URL = "https://api.github.com/search/issues"

rows = []


for start_date, end_date in DATE_RANGES:

    page = 1
    range_count = 0

    while True:

        params = {
            "q": (
                f"repo:{OWNER}/{REPO} "
                "is:pr "
                "is:merged "
                'label:"Bug" '
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

        response.raise_for_status()

        data = response.json()
        items = data["items"]

        if not items:
            break

        for item in items:

            rows.append(
                {
                    "pr_number": item["number"],
                    "title": item["title"],
                    "author": item["user"]["login"],
                    "created_at": item["created_at"],
                    "closed_at": item["closed_at"],
                    "labels": [
                        label["name"]
                        for label in item["labels"]
                    ],
                }
            )

        range_count += len(items)

        print(
            f"{start_date[:4]} | "
            f"página {page} | "
            f"{len(items)} PRs coletadas"
        )

        if len(items) < 100:
            break

        page += 1

        time.sleep(0.2)

    print(
        f"Total do período "
        f"{start_date} a {end_date}: "
        f"{range_count}"
    )

    print()


bugfix_df = pd.DataFrame(rows)

bugfix_df = bugfix_df.drop_duplicates(
    subset="pr_number"
)

bugfix_df.to_csv(
    OUTPUT_FILE,
    index=False,
)


print("Coleta concluída.")
print("Total de PRs de Bug:", len(bugfix_df))

print()
print("Primeiros registros:")
print(bugfix_df.head())

print()
print(
    "Requisições restantes:",
    response.headers.get(
        "X-RateLimit-Remaining"
    ),
)

print()
print(
    f"Arquivo salvo em {OUTPUT_FILE}"
)