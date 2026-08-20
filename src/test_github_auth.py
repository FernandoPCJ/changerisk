import os

import requests
from dotenv import load_dotenv


load_dotenv()

token = os.getenv("GITHUB_TOKEN")

if not token:
    raise RuntimeError("GITHUB_TOKEN não encontrado no arquivo .env")

headers = {
    "Accept": "application/vnd.github+json",
    "Authorization": f"Bearer {token}",
    "X-GitHub-Api-Version": "2022-11-28",
}

response = requests.get(
    "https://api.github.com/rate_limit",
    headers=headers,
    timeout=30,
)

response.raise_for_status()

data = response.json()

core = data["resources"]["core"]

print("Autenticação realizada com sucesso.")
print("Limite:", core["limit"])
print("Restantes:", core["remaining"])
print("Utilizadas:", core["used"])