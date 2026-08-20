import os
import time

import pandas as pd
import requests
from dotenv import load_dotenv


OWNER = "pandas-dev"
REPO = "pandas"

INPUT_FILE = "data/raw/pandas_pulls_pilot_100.csv"
OUTPUT_FILE = "data/raw/pandas_pulls_pilot_100_enriched.csv"


load_dotenv()

token = os.getenv("GITHUB_TOKEN")

if not token:
    raise RuntimeError("GITHUB_TOKEN não encontrado no arquivo .env")


session = requests.Session()

session.headers.update(
    {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
    }
)


df = pd.read_csv(INPUT_FILE)

details = []


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

    details.append(
        {
            "pr_number": pr["number"],
            "title": pr["title"],
            "author": pr["user"]["login"],

            "created_at": pr["created_at"],
            "updated_at": pr["updated_at"],
            "closed_at": pr["closed_at"],
            "merged_at": pr["merged_at"],

            "merge_commit_sha": pr["merge_commit_sha"],

            "commits": pr["commits"],
            "changed_files": pr["changed_files"],

            "additions": pr["additions"],
            "deletions": pr["deletions"],
            "code_churn": (
                pr["additions"]
                + pr["deletions"]
            ),

            "comments": pr["comments"],
            "review_comments": pr["review_comments"],

            "labels": [
                label["name"]
                for label in pr["labels"]
            ],

            "sample_year": row["sample_year"],
        }
    )

    print(
        f"PR {pr_number} coletada "
        f"({index + 1}/{len(df)})"
    )

    time.sleep(0.1)


enriched_df = pd.DataFrame(details)

enriched_df.to_csv(
    OUTPUT_FILE,
    index=False,
)


print()
print("Enriquecimento concluído.")
print("Dimensão:", enriched_df.shape)

print()
print("Distribuição por ano:")
print(
    enriched_df["sample_year"]
    .value_counts()
    .sort_index()
)

print()
print(f"Arquivo salvo em {OUTPUT_FILE}")

print()
print(
    "Requisições restantes:",
    response.headers.get("X-RateLimit-Remaining"),
)