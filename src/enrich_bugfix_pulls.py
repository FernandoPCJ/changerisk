import os
import time

import pandas as pd
import requests
from dotenv import load_dotenv


OWNER = "pandas-dev"
REPO = "pandas"

INPUT_FILE = "data/raw/pandas_bugfix_pulls.csv"
OUTPUT_FILE = "data/raw/pandas_bugfix_pulls_enriched.csv"


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


df = pd.read_csv(INPUT_FILE)

rows = []


for index, row in df.iterrows():

    pr_number = int(row["pr_number"])

    url = (
        f"https://api.github.com/repos/"
        f"{OWNER}/{REPO}/pulls/{pr_number}"
    )

    response = session.get(
        url,
        timeout=30,
    )

    response.raise_for_status()

    pr = response.json()

    rows.append(
        {
            "pr_number": pr["number"],
            "title": pr["title"],
            "author": pr["user"]["login"],
            "created_at": pr["created_at"],
            "closed_at": pr["closed_at"],
            "merged_at": pr["merged_at"],
            "merge_commit_sha": pr["merge_commit_sha"],
            "commits": pr["commits"],
            "changed_files": pr["changed_files"],
            "additions": pr["additions"],
            "deletions": pr["deletions"],
            "code_churn": (
                pr["additions"] + pr["deletions"]
            ),
            "labels": [
                label["name"]
                for label in pr["labels"]
            ],
        }
    )

    print(
        f"Bug PR {pr_number} coletada "
        f"({index + 1}/{len(df)})"
    )

    time.sleep(0.05)


enriched_df = pd.DataFrame(rows)

enriched_df.to_csv(
    OUTPUT_FILE,
    index=False,
)


print()
print("Enriquecimento concluído.")
print("Dimensão:", enriched_df.shape)

print()
print(
    "PRs únicas:",
    enriched_df["pr_number"].nunique()
)

print()
print(
    "merged_at ausente:",
    enriched_df["merged_at"].isna().sum()
)

print()
print(
    "Requisições restantes:",
    response.headers.get("X-RateLimit-Remaining")
)

print()
print(
    f"Arquivo salvo em {OUTPUT_FILE}"
)