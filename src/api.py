from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict

from src.inference import ChangeRiskPredictor


# ============================================================
# GLOBAL PREDICTOR
# ============================================================

predictor = None


# ============================================================
# LIFESPAN
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    global predictor

    predictor = ChangeRiskPredictor()

    yield

    predictor = None


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="ChangeRisk API",
    description=(
        "API para priorização relativa de risco "
        "de Pull Requests."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ============================================================
# REQUEST MODEL
# ============================================================

class PredictionRequest(BaseModel):

    model_config = ConfigDict(
        extra="allow"
    )

    commits: float
    changed_files: float
    additions: float
    deletions: float
    pr_duration_hours: float

    production_files_changed: float
    test_files_changed: float
    documentation_files_changed: float
    other_files_changed: float

    added_files: float
    modified_files: float
    deleted_files: float
    renamed_files: float

    churn_per_file: float
    additions_per_file: float
    deletions_per_file: float
    commits_per_file: float

    production_file_ratio: float
    test_file_ratio: float
    test_to_production_ratio: float
    addition_ratio: float


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "ok",
        "model_loaded": (
            predictor is not None
        ),
    }


# ============================================================
# MODEL INFO
# ============================================================

@app.get("/model")
def model_info():

    if predictor is None:

        raise HTTPException(
            status_code=503,
            detail="Modelo não carregado.",
        )


    metadata = predictor.metadata


    return {
        "project": metadata[
            "project"
        ],

        "repository": metadata[
            "repository"
        ],

        "model": metadata[
            "model"
        ],

        "dataset": metadata[
            "dataset"
        ],

        "feature_count": metadata[
            "feature_count"
        ],

        "train_years": metadata[
            "train_years"
        ],

        "operational_threshold": metadata[
            "operational_threshold"
        ],

        "interpretation": (
            "relative_risk_score"
        ),

        "warning": (
            "O score não representa "
            "probabilidade calibrada de defeito."
        ),
    }


# ============================================================
# FEATURES
# ============================================================

@app.get("/features")
def features():

    if predictor is None:

        raise HTTPException(
            status_code=503,
            detail="Modelo não carregado.",
        )


    return {
        "feature_count": len(
            predictor.features
        ),

        "features": predictor.features,
    }


# ============================================================
# PREDICT
# ============================================================

@app.post("/predict")
def predict(
    request: PredictionRequest,
):

    if predictor is None:

        raise HTTPException(
            status_code=503,
            detail="Modelo não carregado.",
        )


    try:

        payload = (
            request.model_dump()
        )


        result = predictor.predict(
            payload
        )[0]


        return {
            "prediction": result,
        }


    except ValueError as error:

        raise HTTPException(
            status_code=422,
            detail=str(
                error
            ),
        ) from error


    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                "Erro interno durante "
                "a inferência."
            ),
        ) from error


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "service": "ChangeRisk API",
        "status": "online",
        "docs": "/docs",
    }