import os
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv


# ============================================================
# CONFIGURAÇÃO
# ============================================================

OWNER = "pandas-dev"
REPO = "pandas"

INPUT_FILE = Path(
    "data/raw/pandas_pulls_full.csv"
)

OUTPUT_FILE = Path(
    "data/raw/pandas_pulls_full_enriched.csv"
)

ERROR_FILE = Path(
    "data/raw/pandas_pulls_full_errors.csv"
)

CHECKPOINT_EVERY = 50

MAX_RETRIES = 5


# ============================================================
# AUTENTICAÇÃO
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
# FUNÇÕES AUXILIARES
# ============================================================

def save_checkpoint(rows):
    """
    Salva os dados já coletados.
    """

    if not rows:
        return

    df = pd.DataFrame(rows)

    df = (
        df
        .drop_duplicates(
            subset="pr_number",
            keep="last",
        )
        .sort_values("pr_number")
        .reset_index(drop=True)
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )


def save_errors(errors):
    """
    Salva erros encontrados durante a coleta.
    """

    if not errors:
        return

    pd.DataFrame(
        errors
    ).to_csv(
        ERROR_FILE,
        index=False,
    )


def wait_for_rate_limit(response):
    """
    Verifica o limite principal da API e aguarda
    automaticamente quando necessário.
    """

    remaining = response.headers.get(
        "X-RateLimit-Remaining"
    )

    reset = response.headers.get(
        "X-RateLimit-Reset"
    )

    if remaining is None:
        return

    remaining = int(remaining)

    if remaining > 10:
        return

    if reset is None:
        wait_seconds = 60
    else:
        reset_time = int(reset)

        wait_seconds = max(
            reset_time - int(time.time()) + 5,
            5,
        )

    print()
    print(
        f"Limite da API próximo do fim "
        f"({remaining} restantes)."
    )

    print(
        f"Aguardando {wait_seconds} segundos "
        "até a próxima janela..."
    )

    print()

    time.sleep(
        wait_seconds
    )


def get_pull_request(pr_number):
    """
    Busca os detalhes completos de uma PR,
    com tentativas automáticas em caso de erro.
    """

    url = (
        f"https://api.github.com/repos/"
        f"{OWNER}/{REPO}/pulls/{pr_number}"
    )

    for attempt in range(
        1,
        MAX_RETRIES + 1,
    ):

        try:

            response = session.get(
                url,
                timeout=30,
            )

            # --------------------------------------------
            # RATE LIMIT / ABUSE LIMIT
            # --------------------------------------------

            if response.status_code in (
                403,
                429,
            ):

                retry_after = (
                    response.headers.get(
                        "Retry-After"
                    )
                )

                reset = (
                    response.headers.get(
                        "X-RateLimit-Reset"
                    )
                )

                if retry_after:

                    wait_seconds = (
                        int(retry_after) + 2
                    )

                elif reset:

                    wait_seconds = max(
                        int(reset)
                        - int(time.time())
                        + 5,
                        10,
                    )

                else:

                    wait_seconds = 60

                print(
                    f"Rate limit na PR {pr_number}. "
                    f"Aguardando {wait_seconds}s..."
                )

                time.sleep(
                    wait_seconds
                )

                continue


            response.raise_for_status()

            wait_for_rate_limit(
                response
            )

            return response.json(), response


        except requests.RequestException as exc:

            if attempt == MAX_RETRIES:
                raise

            wait_seconds = (
                attempt * 5
            )

            print(
                f"Erro temporário na PR "
                f"{pr_number}: {exc}"
            )

            print(
                f"Tentativa "
                f"{attempt}/{MAX_RETRIES}. "
                f"Aguardando {wait_seconds}s..."
            )

            time.sleep(
                wait_seconds
            )


    raise RuntimeError(
        f"Não foi possível coletar "
        f"a PR {pr_number}"
    )


# ============================================================
# CARREGAR POPULAÇÃO
# ============================================================

source_df = pd.read_csv(
    INPUT_FILE
)

print(
    "PRs existentes na população:",
    len(source_df),
)


# ============================================================
# RETOMADA AUTOMÁTICA
# ============================================================

if OUTPUT_FILE.exists():

    existing_df = pd.read_csv(
        OUTPUT_FILE
    )

    collected_ids = set(
        existing_df[
            "pr_number"
        ].astype(int)
    )

    rows = existing_df.to_dict(
        orient="records"
    )

    print(
        "Checkpoint encontrado."
    )

    print(
        "PRs já enriquecidas:",
        len(collected_ids),
    )

else:

    collected_ids = set()

    rows = []

    print(
        "Nenhum checkpoint encontrado."
    )

    print(
        "Iniciando coleta do zero."
    )


pending_df = source_df[
    ~source_df[
        "pr_number"
    ].isin(collected_ids)
].copy()


print(
    "PRs ainda pendentes:",
    len(pending_df),
)

print()


# ============================================================
# COLETA
# ============================================================

errors = []

processed_since_checkpoint = 0

total_population = len(
    source_df
)

already_collected = len(
    collected_ids
)


for position, (_, row) in enumerate(
    pending_df.iterrows(),
    start=1,
):

    pr_number = int(
        row["pr_number"]
    )

    absolute_position = (
        already_collected
        + position
    )


    try:

        pr, response = get_pull_request(
            pr_number
        )


        rows.append(
            {
                "pr_number": (
                    pr["number"]
                ),

                "title": (
                    pr["title"]
                ),

                "author": (
                    pr["user"]["login"]
                    if pr.get("user")
                    else None
                ),

                "created_at": (
                    pr["created_at"]
                ),

                "updated_at": (
                    pr["updated_at"]
                ),

                "closed_at": (
                    pr["closed_at"]
                ),

                "merged_at": (
                    pr["merged_at"]
                ),

                "merge_commit_sha": (
                    pr["merge_commit_sha"]
                ),

                "commits": (
                    pr["commits"]
                ),

                "changed_files": (
                    pr["changed_files"]
                ),

                "additions": (
                    pr["additions"]
                ),

                "deletions": (
                    pr["deletions"]
                ),

                "code_churn": (
                    pr["additions"]
                    + pr["deletions"]
                ),

                "comments": (
                    pr["comments"]
                ),

                "review_comments": (
                    pr["review_comments"]
                ),

                "labels": [
                    label["name"]
                    for label
                    in pr["labels"]
                ],

                "collection_year": (
                    row["collection_year"]
                ),

                "collection_month": (
                    row["collection_month"]
                ),
            }
        )


        processed_since_checkpoint += 1


        remaining = (
            response.headers.get(
                "X-RateLimit-Remaining"
            )
        )


        print(
            f"[{absolute_position}/"
            f"{total_population}] "
            f"PR {pr_number} coletada "
            f"| restantes API: {remaining}"
        )


    except Exception as exc:

        print(
            f"[ERRO] PR {pr_number}: "
            f"{exc}"
        )


        errors.append(
            {
                "pr_number": (
                    pr_number
                ),

                "error": (
                    str(exc)
                ),
            }
        )


    # ========================================================
    # CHECKPOINT
    # ========================================================

    if (
        processed_since_checkpoint
        >= CHECKPOINT_EVERY
    ):

        save_checkpoint(
            rows
        )

        save_errors(
            errors
        )

        print()
        print(
            f"CHECKPOINT SALVO "
            f"({len(rows)} PRs)"
        )
        print()

        processed_since_checkpoint = 0


    # Pequena pausa para evitar carga agressiva.
    time.sleep(0.1)


# ============================================================
# SALVAR RESULTADO FINAL
# ============================================================

save_checkpoint(
    rows
)

save_errors(
    errors
)


final_df = pd.read_csv(
    OUTPUT_FILE
)


# ============================================================
# VALIDAÇÕES
# ============================================================

print()
print("=" * 60)
print("ENRIQUECIMENTO COMPLETO")
print("=" * 60)

print(
    "PRs na população:",
    total_population,
)

print(
    "PRs enriquecidas:",
    len(final_df),
)

print(
    "PRs únicas:",
    final_df[
        "pr_number"
    ].nunique(),
)

print(
    "Duplicatas:",
    final_df.duplicated(
        subset="pr_number"
    ).sum(),
)

print(
    "merged_at ausente:",
    final_df[
        "merged_at"
    ].isna().sum(),
)

print(
    "merge_commit_sha ausente:",
    final_df[
        "merge_commit_sha"
    ].isna().sum(),
)

print(
    "Erros registrados:",
    len(errors),
)

print()

print(
    "Distribuição por ano:"
)

print(
    final_df[
        "collection_year"
    ]
    .value_counts()
    .sort_index()
)

print()

print(
    f"Arquivo final:\n"
    f"{OUTPUT_FILE}"
)

if errors:

    print()

    print(
        f"Arquivo de erros:\n"
        f"{ERROR_FILE}"
    )