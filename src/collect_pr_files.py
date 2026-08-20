import os
import time

import pandas as pd
import requests
from dotenv import load_dotenv


OWNER = "pandas-dev"
REPO = "pandas"

INPUT_FILE = "data/raw/pandas_pulls_pilot_100_enriched.csv"
OUTPUT_FILE = "data/raw/pandas_pilot_pr_files.csv"


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


prs = pd.read_csv(INPUT_FILE)

file_rows = []


for index, row in prs.iterrows():

    pr_number = int(row["pr_number"])

    page = 1
    total_files = 0

    while True:

        url = (
            f"https://api.github.com/repos/"
            f"{OWNER}/{REPO}/pulls/{pr_number}/files"
        )

        response = session.get(
            url,
            params={
                "per_page": 100,
                "page": page,
            },
            timeout=30,
        )

        response.raise_for_status()

        files = response.json()

        if not files:
            break

        for file in files:

            file_rows.append(
                {
                    "pr_number": pr_number,
                    "sample_year": int(
                        row["sample_year"]
                    ),
                    "filename": file["filename"],
                    "status": file["status"],
                    "additions": file["additions"],
                    "deletions": file["deletions"],
                    "changes": file["changes"],
                    "previous_filename": (
                        file.get("previous_filename")
                    ),
                }
            )

        total_files += len(files)

        if len(files) < 100:
            break

        page += 1

    print(
        f"PR {pr_number}: "
        f"{total_files} arquivos "
        f"({index + 1}/{len(prs)})"
    )

    time.sleep(0.05)


files_df = pd.DataFrame(file_rows)

files_df.to_csv(
    OUTPUT_FILE,
    index=False,
)


print()
print("Coleta de arquivos concluída.")
print("Dimensão:", files_df.shape)

print()
print(
    "PRs representadas:",
    files_df["pr_number"].nunique()
)

print()
print("Exemplo:")
print(files_df.head(10))

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