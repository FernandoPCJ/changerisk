from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

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
        extra="forbid"
    )

    commits: float = Field(ge=0)
    changed_files: float = Field(ge=0)

    additions: float = Field(ge=0)
    deletions: float = Field(ge=0)

    pr_duration_hours: float = Field(ge=0)

    production_files_changed: float = Field(ge=0)
    test_files_changed: float = Field(ge=0)
    documentation_files_changed: float = Field(ge=0)
    other_files_changed: float = Field(ge=0)

    added_files: float = Field(ge=0)
    modified_files: float = Field(ge=0)
    deleted_files: float = Field(ge=0)
    renamed_files: float = Field(ge=0)

    churn_per_file: float = Field(ge=0)
    additions_per_file: float = Field(ge=0)
    deletions_per_file: float = Field(ge=0)
    commits_per_file: float = Field(ge=0)

    production_file_ratio: float = Field(
        ge=0,
        le=1,
    )

    test_file_ratio: float = Field(
        ge=0,
        le=1,
    )

    test_to_production_ratio: float = Field(
        ge=0
    )

    addition_ratio: float = Field(
        ge=0,
        le=1,
    )


# ============================================================
# RESPONSE MODELS
# ============================================================

class HealthResponse(BaseModel):

    status: Literal["ok"]

    model_loaded: bool


class ModelInfoResponse(BaseModel):

    project: str

    repository: str

    model: str

    dataset: str

    feature_count: int

    train_years: list[int]

    operational_threshold: float

    interpretation: Literal[
        "relative_risk_score"
    ]

    warning: str


class FeaturesResponse(BaseModel):

    feature_count: int

    features: list[str]


class PredictionResult(BaseModel):

    risk_score: float

    risk_level: Literal[
        "normal",
        "elevated",
        "high",
        "very_high",
    ]

    operational_flag: bool

    operational_threshold: float

    interpretation: Literal[
        "relative_risk_score"
    ]


class PredictionResponse(BaseModel):

    prediction: PredictionResult


class RootResponse(BaseModel):

    service: str

    status: Literal["online"]

    docs: str


# ============================================================
# HEALTH
# ============================================================

@app.get(
    "/health",
    response_model=HealthResponse,
)
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

@app.get(
    "/model",
    response_model=ModelInfoResponse,
)
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

@app.get(
    "/features",
    response_model=FeaturesResponse,
)
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

@app.post(
    "/predict",
    response_model=PredictionResponse,
)
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

@app.get(
    "/",
    response_model=RootResponse,
)
def root():

    return {
        "service": "ChangeRisk API",

        "status": "online",

        "docs": "/docs",
    }