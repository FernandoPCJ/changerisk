from pathlib import Path
import json

import mlflow
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

EXPERIMENT_NAME = "ChangeRisk"

RUN_NAME = "logistic_structural_final"


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = (
    ROOT
    / "data"
    / "processed"
)

MODELS_DIR = (
    ROOT
    / "models"
)

MLRUNS_DIR = (
    ROOT
    / "mlruns"
)

MLFLOW_DB = (
    ROOT
    / "mlflow.db"
)


MODEL_FILE = (
    MODELS_DIR
    / "changerisk_logistic_structural.joblib"
)

METADATA_FILE = (
    MODELS_DIR
    / "changerisk_logistic_structural_metadata.json"
)

MODEL_COMPARISON_FILE = (
    PROCESSED_DIR
    / "pandas_model_comparison_2023.csv"
)

RANKING_FILE = (
    PROCESSED_DIR
    / "pandas_ranking_evaluation_2023.csv"
)

BOOTSTRAP_SUMMARY_FILE = (
    PROCESSED_DIR
    / "pandas_bootstrap_model_summary.csv"
)

THRESHOLD_FILE = (
    PROCESSED_DIR
    / "pandas_logistic_threshold_summary_2023.csv"
)

FINAL_TEST_FILE = (
    PROCESSED_DIR
    / "pandas_final_temporal_test.csv"
)

COEFFICIENTS_FILE = (
    PROCESSED_DIR
    / "pandas_logistic_coefficients.csv"
)

SCORE_DISTRIBUTION_FILE = (
    PROCESSED_DIR
    / "pandas_score_distribution_by_year.csv"
)

DRIFT_FILE = (
    PROCESSED_DIR
    / "pandas_temporal_drift_diagnostics.csv"
)


# ============================================================
# HELPERS
# ============================================================

def validate_required_files(paths):

    missing = [
        path
        for path in paths
        if not path.exists()
    ]

    if not missing:
        return

    print()
    print("=" * 76)
    print("ARQUIVOS OBRIGATÓRIOS AUSENTES")
    print("=" * 76)

    for path in missing:
        print("-", path)

    raise FileNotFoundError(
        "Arquivos necessários para registrar "
        "o experimento no MLflow não foram encontrados."
    )


def require_row(
    dataframe,
    condition,
    description,
):
    """
    Retorna exatamente a primeira linha correspondente
    ao critério informado e gera uma mensagem clara caso
    nenhuma linha seja encontrada.
    """

    result = (
        dataframe[
            condition
        ]
        .copy()
    )

    if result.empty:

        raise ValueError(
            f"Nenhum registro encontrado para: "
            f"{description}"
        )

    return result.iloc[0]


# ============================================================
# VALIDATE REQUIRED FILES
# ============================================================

required_files = [
    MODEL_FILE,
    METADATA_FILE,
    MODEL_COMPARISON_FILE,
    RANKING_FILE,
    THRESHOLD_FILE,
    FINAL_TEST_FILE,
]


validate_required_files(
    required_files
)


# ============================================================
# LOAD METADATA
# ============================================================

with open(
    METADATA_FILE,
    "r",
    encoding="utf-8",
) as file:

    metadata = json.load(
        file
    )


# ============================================================
# VALIDATION — 2023
# ============================================================

comparison = pd.read_csv(
    MODEL_COMPARISON_FILE
)


validation_result = require_row(
    dataframe=comparison,

    condition=(
        (
            comparison["dataset"]
            == "structural"
        )
        &
        (
            comparison["model"]
            == "logistic_balanced"
        )
    ),

    description=(
        "dataset structural + "
        "modelo logistic_balanced"
    ),
)


# ============================================================
# RANKING — 2023
# ============================================================

ranking = pd.read_csv(
    RANKING_FILE
)


logistic_ranking = (
    ranking[
        ranking["model"]
        == "logistic_balanced"
    ]
    .copy()
)


if logistic_ranking.empty:

    raise ValueError(
        "Nenhum resultado de ranking encontrado "
        "para logistic_balanced."
    )


ranking_5 = require_row(
    dataframe=logistic_ranking,

    condition=(
        logistic_ranking[
            "top_pct"
        ] == 5
    ),

    description=(
        "logistic_balanced no top 5%"
    ),
)


ranking_10 = require_row(
    dataframe=logistic_ranking,

    condition=(
        logistic_ranking[
            "top_pct"
        ] == 10
    ),

    description=(
        "logistic_balanced no top 10%"
    ),
)


# ============================================================
# THRESHOLD — 2023
# ============================================================

thresholds = pd.read_csv(
    THRESHOLD_FILE
)


threshold_result = require_row(
    dataframe=thresholds,

    condition=(
        thresholds[
            "criterion"
        ] == "max_mcc"
    ),

    description=(
        "threshold selecionado por max_mcc"
    ),
)


# ============================================================
# FINAL OUT-OF-TIME TEST — 2024/2025
# ============================================================

final_test = pd.read_csv(
    FINAL_TEST_FILE
)


if final_test.empty:

    raise ValueError(
        "O arquivo do teste temporal final "
        "não possui resultados."
    )


final_result = (
    final_test
    .iloc[0]
)


# ============================================================
# MLFLOW CONFIG
# ============================================================

MLRUNS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# MLflow 3.x:
# - SQLite guarda experimentos, runs, parâmetros,
#   métricas e tags.
# - mlruns/ guarda os artefatos físicos.

tracking_uri = (
    f"sqlite:///"
    f"{MLFLOW_DB.resolve().as_posix()}"
)


artifact_location = (
    MLRUNS_DIR
    .resolve()
    .as_uri()
)


mlflow.set_tracking_uri(
    tracking_uri
)


# ============================================================
# CREATE / FIND EXPERIMENT
# ============================================================

experiment = (
    mlflow.get_experiment_by_name(
        EXPERIMENT_NAME
    )
)


if experiment is None:

    experiment_id = (
        mlflow.create_experiment(
            name=EXPERIMENT_NAME,
            artifact_location=(
                artifact_location
            ),
        )
    )

    experiment = (
        mlflow.get_experiment(
            experiment_id
        )
    )

else:

    experiment_id = (
        experiment.experiment_id
    )


print()
print("=" * 76)
print("CONFIGURAÇÃO DO MLFLOW")
print("=" * 76)

print(
    "Tracking DB:",
    MLFLOW_DB,
)

print(
    "Tracking URI:",
    tracking_uri,
)

print(
    "Artefatos:",
    artifact_location,
)

print(
    "Experiment ID:",
    experiment_id,
)


# ============================================================
# START RUN
# ============================================================

with mlflow.start_run(
    experiment_id=experiment_id,
    run_name=RUN_NAME,
) as run:

    # ========================================================
    # TAGS
    # ========================================================

    mlflow.set_tags(
        {
            "project": (
                "ChangeRisk"
            ),

            "repository": (
                "pandas-dev/pandas"
            ),

            "task": (
                "defect-risk-ranking"
            ),

            "experiment_status": (
                "finalized"
            ),

            "validation_strategy": (
                "temporal"
            ),

            "test_strategy": (
                "out-of-time-stress-test"
            ),

            "score_interpretation": (
                "relative-risk-score"
            ),
        }
    )


    # ========================================================
    # PARAMETERS
    # ========================================================

    mlflow.log_params(
        {
            "model": (
                metadata[
                    "model"
                ]
            ),

            "dataset": (
                metadata[
                    "dataset"
                ]
            ),

            "feature_count": (
                metadata[
                    "feature_count"
                ]
            ),

            "scaler": (
                metadata[
                    "scaler"
                ]
            ),

            "class_weight": (
                metadata[
                    "class_weight"
                ]
            ),

            "random_state": (
                metadata[
                    "random_state"
                ]
            ),

            "train_years": ",".join(
                str(year)
                for year in metadata[
                    "train_years"
                ]
            ),

            "validation_year": (
                metadata[
                    "validation_year"
                ]
            ),

            "out_of_time_years": ",".join(
                str(year)
                for year in metadata[
                    "out_of_time_test_years"
                ]
            ),

            "operational_threshold": (
                metadata[
                    "operational_threshold"
                ]
            ),

            "training_rows": (
                metadata[
                    "training_rows"
                ]
            ),

            "training_positives": (
                metadata[
                    "training_positives"
                ]
            ),

            "training_negatives": (
                metadata[
                    "training_negatives"
                ]
            ),
        }
    )


    # ========================================================
    # VALIDATION METRICS — 2023
    # ========================================================

    mlflow.log_metrics(
        {
            "validation_pr_auc": float(
                validation_result[
                    "pr_auc"
                ]
            ),

            "validation_pr_auc_lift": float(
                validation_result[
                    "pr_auc_lift"
                ]
            ),

            "validation_roc_auc": float(
                validation_result[
                    "roc_auc"
                ]
            ),

            "validation_mcc_threshold_05": float(
                validation_result[
                    "mcc_05"
                ]
            ),

            "validation_recall_top_5pct": float(
                ranking_5[
                    "recall_at_k"
                ]
            ),

            "validation_precision_top_5pct": float(
                ranking_5[
                    "precision_at_k"
                ]
            ),

            "validation_lift_top_5pct": float(
                ranking_5[
                    "lift_at_k"
                ]
            ),

            "validation_recall_top_10pct": float(
                ranking_10[
                    "recall_at_k"
                ]
            ),

            "validation_lift_top_10pct": float(
                ranking_10[
                    "lift_at_k"
                ]
            ),
        }
    )


    # ========================================================
    # OPERATIONAL THRESHOLD — 2023
    # ========================================================

    mlflow.log_metrics(
        {
            "threshold_validation_mcc": float(
                threshold_result[
                    "mcc"
                ]
            ),

            "threshold_validation_precision": float(
                threshold_result[
                    "precision"
                ]
            ),

            "threshold_validation_recall": float(
                threshold_result[
                    "recall"
                ]
            ),

            "threshold_validation_flagged_pct": float(
                threshold_result[
                    "flagged_pct"
                ]
            ),
        }
    )


    # ========================================================
    # FINAL OUT-OF-TIME TEST — 2024/2025
    # ========================================================

    mlflow.log_metrics(
        {
            "oot_prevalence": float(
                final_result[
                    "prevalence"
                ]
            ),

            "oot_pr_auc": float(
                final_result[
                    "pr_auc"
                ]
            ),

            "oot_pr_auc_lift": float(
                final_result[
                    "pr_auc_lift"
                ]
            ),

            "oot_roc_auc": float(
                final_result[
                    "roc_auc"
                ]
            ),

            "oot_precision": float(
                final_result[
                    "precision"
                ]
            ),

            "oot_recall": float(
                final_result[
                    "recall"
                ]
            ),

            "oot_f1": float(
                final_result[
                    "f1"
                ]
            ),

            "oot_balanced_accuracy": float(
                final_result[
                    "balanced_accuracy"
                ]
            ),

            "oot_mcc": float(
                final_result[
                    "mcc"
                ]
            ),

            "oot_flagged_pct": float(
                final_result[
                    "flagged_pct"
                ]
            ),
        }
    )


    # ========================================================
    # MODEL ARTIFACTS
    # ========================================================

    mlflow.log_artifact(
        str(
            MODEL_FILE
        ),
        artifact_path="model",
    )


    mlflow.log_artifact(
        str(
            METADATA_FILE
        ),
        artifact_path="model",
    )


    # ========================================================
    # ANALYSIS ARTIFACTS
    # ========================================================

    analysis_files = [
        MODEL_COMPARISON_FILE,
        RANKING_FILE,
        BOOTSTRAP_SUMMARY_FILE,
        THRESHOLD_FILE,
        FINAL_TEST_FILE,
        COEFFICIENTS_FILE,
        SCORE_DISTRIBUTION_FILE,
        DRIFT_FILE,
    ]


    for artifact in analysis_files:

        if artifact.exists():

            mlflow.log_artifact(
                str(
                    artifact
                ),
                artifact_path="analysis",
            )


    # ========================================================
    # FEATURE SCHEMA
    # ========================================================

    feature_file = (
        ROOT
        / "mlflow_feature_list.json"
    )


    try:

        with open(
            feature_file,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                metadata[
                    "features"
                ],
                file,
                indent=2,
                ensure_ascii=False,
            )


        mlflow.log_artifact(
            str(
                feature_file
            ),
            artifact_path="schema",
        )

    finally:

        if feature_file.exists():

            feature_file.unlink()


    # ========================================================
    # RESULT
    # ========================================================

    print()
    print("=" * 76)
    print("EXPERIMENTO REGISTRADO NO MLFLOW")
    print("=" * 76)

    print(
        "Experiment:",
        EXPERIMENT_NAME,
    )

    print(
        "Experiment ID:",
        experiment_id,
    )

    print(
        "Run name:",
        RUN_NAME,
    )

    print(
        "Run ID:",
        run.info.run_id,
    )

    print()

    print(
        "Tracking URI:",
        mlflow.get_tracking_uri(),
    )

    print(
        "Artifact URI:",
        run.info.artifact_uri,
    )

    print()

    print(
        "Modelo:",
        metadata[
            "model"
        ],
    )

    print(
        "Dataset:",
        metadata[
            "dataset"
        ],
    )

    print(
        "Features:",
        metadata[
            "feature_count"
        ],
    )

    print(
        "Threshold:",
        metadata[
            "operational_threshold"
        ],
    )

    print()

    print(
        "Validação 2023 PR-AUC:",
        round(
            float(
                validation_result[
                    "pr_auc"
                ]
            ),
            5,
        ),
    )

    print(
        "Teste OOT PR-AUC:",
        round(
            float(
                final_result[
                    "pr_auc"
                ]
            ),
            5,
        ),
    )

    print()
    print(
        "Registro concluído."
    )