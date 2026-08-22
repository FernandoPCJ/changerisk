
import pandas as pd
import pytest

from src.inference import ChangeRiskPredictor



# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture(scope="module")
def predictor():

    return ChangeRiskPredictor()


@pytest.fixture(scope="module")
def valid_sample(
    predictor,
):

    sample = {
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

    dataframe = pd.DataFrame(
        [sample]
    )

    return dataframe[
        predictor.features
    ].copy()


# ============================================================
# MODEL / METADATA
# ============================================================

def test_model_loads(
    predictor,
):

    assert predictor.model is not None


def test_expected_feature_count(
    predictor,
):

    assert len(
        predictor.features
    ) == 21


def test_threshold(
    predictor,
):

    assert predictor.threshold == pytest.approx(
        0.69
    )


# ============================================================
# SINGLE PREDICTION
# ============================================================

def test_single_prediction(
    predictor,
    valid_sample,
):

    result = predictor.predict(
        valid_sample
    )

    assert len(result) == 1

    prediction = result[0]

    expected_keys = {
        "risk_score",
        "risk_level",
        "operational_flag",
        "operational_threshold",
        "interpretation",
    }

    assert (
        set(
            prediction.keys()
        )
        == expected_keys
    )


# ============================================================
# SCORE VALIDATION
# ============================================================

def test_score_range(
    predictor,
    valid_sample,
):

    prediction = predictor.predict(
        valid_sample
    )[0]

    assert (
        0.0
        <= prediction["risk_score"]
        <= 1.0
    )


def test_interpretation(
    predictor,
    valid_sample,
):

    prediction = predictor.predict(
        valid_sample
    )[0]

    assert (
        prediction[
            "interpretation"
        ]
        == "relative_risk_score"
    )


# ============================================================
# RISK LEVEL
# ============================================================

def test_risk_level_is_valid(
    predictor,
    valid_sample,
):

    prediction = predictor.predict(
        valid_sample
    )[0]

    valid_levels = {
        "normal",
        "elevated",
        "high",
        "very_high",
    }

    assert (
        prediction[
            "risk_level"
        ]
        in valid_levels
    )


# ============================================================
# OPERATIONAL FLAG
# ============================================================

def test_operational_flag_matches_threshold(
    predictor,
    valid_sample,
):

    prediction = predictor.predict(
        valid_sample
    )[0]

    expected = (
        prediction[
            "risk_score"
        ]
        >= predictor.threshold
    )

    assert (
        prediction[
            "operational_flag"
        ]
        == expected
    )


# ============================================================
# BATCH
# ============================================================

def test_batch_prediction(
    predictor,
    valid_sample,
):

    sample = pd.concat(
        [
            valid_sample
            for _ in range(10)
        ],
        ignore_index=True,
    )

    result = predictor.predict(
        sample
    )

    assert len(result) == 10


# ============================================================
# DICT INPUT
# ============================================================

def test_dict_input(
    predictor,
    valid_sample,
):

    sample_dict = (
        valid_sample
        .iloc[0]
        .to_dict()
    )

    result = predictor.predict(
        sample_dict
    )

    assert len(result) == 1


# ============================================================
# EXTRA COLUMN
# ============================================================

def test_extra_columns_are_ignored(
    predictor,
    valid_sample,
):

    sample = (
        valid_sample.copy()
    )

    sample[
        "unused_column"
    ] = 123


    result = predictor.predict(
        sample
    )

    assert len(result) == 1


# ============================================================
# MISSING FEATURE
# ============================================================

def test_missing_feature_raises_error(
    predictor,
    valid_sample,
):

    missing_feature = (
        predictor.features[0]
    )


    invalid = (
        valid_sample
        .drop(
            columns=[
                missing_feature
            ]
        )
    )


    with pytest.raises(
        ValueError,
        match="Features ausentes",
    ):

        predictor.predict(
            invalid
        )


# ============================================================
# NON-NUMERIC VALUE
# ============================================================

def test_non_numeric_value_raises_error(
    predictor,
    valid_sample,
):

    # Converte a amostra para dict para permitir inserir
    # propositalmente um valor não numérico.
    invalid = (
        valid_sample
        .iloc[0]
        .to_dict()
    )

    feature = predictor.features[0]

    invalid[feature] = "valor_invalido"

    with pytest.raises(
        ValueError,
        match="Valores ausentes ou não numéricos",
    ):

        predictor.predict(
            invalid
        )

# ============================================================
# INVALID INPUT TYPE
# ============================================================

def test_invalid_input_type(
    predictor,
):

    with pytest.raises(
        TypeError
    ):

        predictor.predict(
            "entrada inválida"
        )