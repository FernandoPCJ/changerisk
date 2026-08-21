from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

MODEL_FILE = (
    ROOT
    / "models"
    / "changerisk_logistic_structural.joblib"
)

METADATA_FILE = (
    ROOT
    / "models"
    / "changerisk_logistic_structural_metadata.json"
)


# ============================================================
# PREDICTOR
# ============================================================

class ChangeRiskPredictor:

    def __init__(
        self,
        model_file=MODEL_FILE,
        metadata_file=METADATA_FILE,
    ):
        """
        Carrega o modelo ChangeRisk e seus metadados.

        O score produzido pelo modelo representa risco relativo.
        Ele NÃO deve ser interpretado como probabilidade
        calibrada de defeito.
        """

        self.model_file = Path(
            model_file
        )

        self.metadata_file = Path(
            metadata_file
        )


        if not self.model_file.exists():

            raise FileNotFoundError(
                f"Modelo não encontrado: "
                f"{self.model_file}"
            )


        if not self.metadata_file.exists():

            raise FileNotFoundError(
                f"Metadados não encontrados: "
                f"{self.metadata_file}"
            )


        self.model = joblib.load(
            self.model_file
        )


        with open(
            self.metadata_file,
            "r",
            encoding="utf-8",
        ) as file:

            self.metadata = json.load(
                file
            )


        self.features = (
            self.metadata[
                "features"
            ]
        )


        self.threshold = float(
            self.metadata[
                "operational_threshold"
            ]
        )


        self.score_reference = (
            self.metadata[
                "score_reference"
            ]
        )


    # ========================================================
    # VALIDATION
    # ========================================================

    def validate_features(
        self,
        data,
    ):
        """
        Valida e organiza as features na mesma ordem
        utilizada durante o treinamento.
        """

        if isinstance(
            data,
            dict,
        ):

            data = pd.DataFrame(
                [data]
            )


        elif isinstance(
            data,
            pd.Series,
        ):

            data = pd.DataFrame(
                [data.to_dict()]
            )


        elif isinstance(
            data,
            pd.DataFrame,
        ):

            data = data.copy()


        else:

            raise TypeError(
                "Entrada deve ser dict, "
                "pandas.Series ou pandas.DataFrame."
            )


        missing = [
            feature
            for feature in self.features
            if feature not in data.columns
        ]


        if missing:

            raise ValueError(
                "Features ausentes: "
                + ", ".join(
                    missing
                )
            )


        # Remove colunas extras e força a ordem correta.
        data = data[
            self.features
        ].copy()


        for feature in self.features:

            data[
                feature
            ] = pd.to_numeric(
                data[
                    feature
                ],
                errors="coerce",
            )


        missing_values = (
            data
            .isna()
            .sum()
        )


        invalid_columns = (
            missing_values[
                missing_values > 0
            ]
            .index
            .tolist()
        )


        if invalid_columns:

            raise ValueError(
                "Valores ausentes ou não numéricos "
                "nas features: "
                + ", ".join(
                    invalid_columns
                )
            )


        matrix = data.to_numpy(
            dtype=float
        )


        if not np.isfinite(
            matrix
        ).all():

            raise ValueError(
                "A entrada contém valores "
                "infinitos ou inválidos."
            )


        return data


    # ========================================================
    # RISK LEVEL
    # ========================================================

    def risk_level(
        self,
        score,
    ):
        """
        Classifica o score relativamente à distribuição
        observada no período de desenvolvimento.
        """

        p75 = float(
            self.score_reference[
                "p75"
            ]
        )

        p90 = float(
            self.score_reference[
                "p90"
            ]
        )

        p95 = float(
            self.score_reference[
                "p95"
            ]
        )


        if score >= p95:

            return "very_high"


        if score >= p90:

            return "high"


        if score >= p75:

            return "elevated"


        return "normal"


    # ========================================================
    # PREDICT
    # ========================================================

    def predict(
        self,
        data,
    ):
        """
        Retorna scores relativos de risco.

        Não interpreta os scores como probabilidades
        calibradas de defeito.
        """

        X = self.validate_features(
            data
        )


        scores = (
            self.model
            .predict_proba(
                X
            )[:, 1]
        )


        results = []


        for score in scores:

            score = float(
                score
            )


            results.append(
                {
                    "risk_score": (
                        round(
                            score,
                            6,
                        )
                    ),

                    "risk_level": (
                        self.risk_level(
                            score
                        )
                    ),

                    "operational_flag": (
                        bool(
                            score
                            >= self.threshold
                        )
                    ),

                    "operational_threshold": (
                        self.threshold
                    ),

                    "interpretation": (
                        "relative_risk_score"
                    ),
                }
            )


        return results


# ============================================================
# LOCAL TEST
# ============================================================

if __name__ == "__main__":

    DATA_FILE = (
        ROOT
        / "data"
        / "processed"
        / "pandas_ml_structural.csv"
    )


    dataset = pd.read_csv(
        DATA_FILE
    )


    predictor = (
        ChangeRiskPredictor()
    )


    print()
    print("=" * 72)
    print("TESTE DA CAMADA DE INFERÊNCIA")
    print("=" * 72)


    # Escolhemos algumas PRs apenas para testar
    # o fluxo técnico de inferência.

    sample = (
        dataset
        .head(5)
        .copy()
    )


    predictions = (
        predictor.predict(
            sample
        )
    )


    output = sample[
        [
            "pr_number",
            "collection_year",
            "observed_defect_90d",
        ]
    ].copy()


    prediction_df = pd.DataFrame(
        predictions
    )


    output = pd.concat(
        [
            output.reset_index(
                drop=True
            ),
            prediction_df,
        ],
        axis=1,
    )


    print(
        output.to_string(
            index=False
        )
    )


    print()
    print(
        "Features esperadas:",
        len(
            predictor.features
        ),
    )


    print(
        "Threshold operacional:",
        predictor.threshold,
    )


    print()
    print(
        "ATENÇÃO:"
    )

    print(
        "risk_score não representa "
        "probabilidade calibrada de defeito."
    )