import os

import pandas as pd
import requests
from dotenv import load_dotenv


OWNER = "pandas-dev"
REPO = "pandas"

YEARS = [2022, 2023, 2024, 2025]
PRS_PER_YEAR = 25

OUTPUT_FILE = "data/raw/pandas_pulls_pilot_100.csv"

SEARCH_URL = "https://api.github.com/search/issues"


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


pull_requests = []


for year in YEARS:

    params = {
        "q": (
            f"repo:{OWNER}/{REPO} "
            "is:pr "
            "is:merged "
            f"merged:{year}-01-01..{year}-12-31"
        ),
        "sort": "created",
        "order": "desc",
        "per_page": PRS_PER_YEAR,
    }

    response = session.get(
        SEARCH_URL,
        params=params,
        timeout=30,
    )

    response.raise_for_status()

    data = response.json()

    print(
        f"{year}: "
        f"{data['total_count']} PRs encontradas | "
        f"{len(data['items'])} coletadas para o piloto"
    )

    for item in data["items"]:

        pull_requests.append(
            {
                "pr_number": item["number"],
                "title": item["title"],
                "author": item["user"]["login"],
                "created_at": item["created_at"],
                "closed_at": item["closed_at"],
                "comments": item["comments"],
                "labels": [
                    label["name"]
                    for label in item["labels"]
                ],
                "sample_year": year,
            }
        )


df = pd.DataFrame(pull_requests)

df.to_csv(
    OUTPUT_FILE,
    index=False,
)


print()
print("Coleta concluída.")
print("Total de PRs no piloto:", len(df))

print()
print("Distribuição por ano:")
print(df["sample_year"].value_counts().sort_index())

print()
print(f"Arquivo salvo em {OUTPUT_FILE}")