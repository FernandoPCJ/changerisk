import pytest

from fastapi.testclient import TestClient

from src.api import app


# ============================================================
# PAYLOADS
# ============================================================

LOW_RISK_PAYLOAD = {
    "commits": 62.0,
    "changed_files": 4.0,
    "additions": 225.0,
    "deletions": 6.0,
    "pr_duration_hours": 22096.657777777778,
    "production_files_changed": 2.0,
    "test_files_changed": 1.0,
    "documentation_files_changed": 1.0,
    "other_files_changed": 0.0,
    "added_files": 0.0,
    "modified_files": 4.0,
    "deleted_files": 0.0,
    "renamed_files": 0.0,
    "churn_per_file": 57.75,
    "additions_per_file": 56.25,
    "deletions_per_file": 1.5,
    "commits_per_file": 15.5,
    "production_file_ratio": 0.5,
    "test_file_ratio": 0.25,
    "test_to_production_ratio": 0.5,
    "addition_ratio": 0.974025974025974,
}


HIGH_RISK_PAYLOAD = {
    "commits": 7.0,
    "changed_files": 28.0,
    "additions": 8307.0,
    "deletions": 8354.0,
    "pr_duration_hours": 888.3330555555556,
    "production_files_changed": 26.0,
    "test_files_changed": 0.0,
    "documentation_files_changed": 1.0,
    "other_files_changed": 1.0,
    "added_files": 0.0,
    "modified_files": 28.0,
    "deleted_files": 0.0,
    "renamed_files": 0.0,
    "churn_per_file": 595.0357142857143,
    "additions_per_file": 296.67857142857144,
    "deletions_per_file": 298.35714285714283,
    "commits_per_file": 0.25,
    "production_file_ratio": 0.9285714285714286,
    "test_file_ratio": 0.0,
    "test_to_production_ratio": 0.0,
    "addition_ratio": 0.4985895204369485,
}


# ============================================================
# CLIENT
# ============================================================

@pytest.fixture(scope="module")
def client():

    with TestClient(
        app
    ) as test_client:

        yield test_client


# ============================================================
# ROOT
# ============================================================

def test_root(
    client,
):

    response = client.get(
        "/"
    )

    assert response.status_code == 200

    body = response.json()

    assert (
        body["service"]
        == "ChangeRisk API"
    )

    assert (
        body["status"]
        == "online"
    )

    assert (
        body["docs"]
        == "/docs"
    )


# ============================================================
# HEALTH
# ============================================================

def test_health(
    client,
):

    response = client.get(
        "/health"
    )

    assert response.status_code == 200

    body = response.json()

    assert body == {
        "status": "ok",
        "model_loaded": True,
    }


# ============================================================
# MODEL INFO
# ============================================================

def test_model_info(
    client,
):

    response = client.get(
        "/model"
    )

    assert response.status_code == 200

    body = response.json()

    assert (
        body["project"]
        == "ChangeRisk"
    )

    assert (
        body["model"]
        == "LogisticRegression"
    )

    assert (
        body["dataset"]
        == "structural"
    )

    assert (
        body["feature_count"]
        == 21
    )

    assert (
        body["train_years"]
        == [2022, 2023]
    )

    assert (
        body[
            "operational_threshold"
        ]
        == pytest.approx(
            0.69
        )
    )


# ============================================================
# FEATURES
# ============================================================

def test_features(
    client,
):

    response = client.get(
        "/features"
    )

    assert response.status_code == 200

    body = response.json()

    assert (
        body["feature_count"]
        == 21
    )

    assert (
        len(
            body["features"]
        )
        == 21
    )

    assert (
        "changed_files"
        in body["features"]
    )

    assert (
        "addition_ratio"
        in body["features"]
    )


# ============================================================
# LOW RISK
# ============================================================

def test_predict_low_risk(
    client,
):

    response = client.post(
        "/predict",
        json=LOW_RISK_PAYLOAD,
    )

    assert response.status_code == 200

    prediction = (
        response
        .json()[
            "prediction"
        ]
    )

    assert (
        0
        <= prediction[
            "risk_score"
        ]
        <= 1
    )

    assert (
        prediction[
            "risk_level"
        ]
        == "normal"
    )

    assert (
        prediction[
            "operational_flag"
        ]
        is False
    )

    assert (
        prediction[
            "operational_threshold"
        ]
        == pytest.approx(
            0.69
        )
    )

    assert (
        prediction[
            "interpretation"
        ]
        == "relative_risk_score"
    )


# ============================================================
# HIGH RISK
# ============================================================

def test_predict_high_risk(
    client,
):

    response = client.post(
        "/predict",
        json=HIGH_RISK_PAYLOAD,
    )

    assert response.status_code == 200

    prediction = (
        response
        .json()[
            "prediction"
        ]
    )

    assert (
        prediction[
            "risk_score"
        ]
        > 0.69
    )

    assert (
        prediction[
            "risk_level"
        ]
        == "very_high"
    )

    assert (
        prediction[
            "operational_flag"
        ]
        is True
    )


# ============================================================
# MISSING FIELD
# ============================================================

def test_missing_field_returns_422(
    client,
):

    invalid = (
        LOW_RISK_PAYLOAD.copy()
    )

    invalid.pop(
        "commits"
    )


    response = client.post(
        "/predict",
        json=invalid,
    )


    assert (
        response.status_code
        == 422
    )


# ============================================================
# EXTRA FIELD
# ============================================================

def test_extra_field_returns_422(
    client,
):

    invalid = (
        LOW_RISK_PAYLOAD.copy()
    )

    invalid[
        "unexpected_feature"
    ] = 123


    response = client.post(
        "/predict",
        json=invalid,
    )


    assert (
        response.status_code
        == 422
    )


# ============================================================
# NEGATIVE VALUE
# ============================================================

def test_negative_value_returns_422(
    client,
):

    invalid = (
        LOW_RISK_PAYLOAD.copy()
    )

    invalid[
        "changed_files"
    ] = -1


    response = client.post(
        "/predict",
        json=invalid,
    )


    assert (
        response.status_code
        == 422
    )


# ============================================================
# INVALID RATIO
# ============================================================

def test_ratio_above_one_returns_422(
    client,
):

    invalid = (
        LOW_RISK_PAYLOAD.copy()
    )

    invalid[
        "addition_ratio"
    ] = 1.5


    response = client.post(
        "/predict",
        json=invalid,
    )


    assert (
        response.status_code
        == 422
    )


# ============================================================
# NON-NUMERIC VALUE
# ============================================================

def test_non_numeric_returns_422(
    client,
):

    invalid = (
        LOW_RISK_PAYLOAD.copy()
    )

    invalid[
        "additions"
    ] = "abc"


    response = client.post(
        "/predict",
        json=invalid,
    )


    assert (
        response.status_code
        == 422
    )