import pytest
import pyspark.pandas as ps

from SparkAutoML.ml_module.BinaryClassifier_module.classifier_file import BClassifier


@pytest.fixture
def sample_data_train():
    df = ps.DataFrame(
        dict(
            col_a=[1, 2, 3, 4],
            col_b=[0.24, 0.56, 0.12, 0.39],
            col_cat=["a", "a", "a", "b"],
            label=[1, 0, 0, 1],
        )
    ).to_spark()
    return df


@pytest.fixture
def sample_data_holdout():
    df = ps.DataFrame(
        dict(
            col_a=[5, 6, 8],
            col_b=[0.30, 0.80, 0.32,],
            col_cat=["a", "a", "b"],
            label=[1, 0, 0,],
        )
    ).to_spark()
    return df


@pytest.fixture
def sample_data_unseen():
    df = ps.DataFrame(
        dict(col_a=[5, 6, 8], col_b=[0.30, 0.80, 0.32,], col_cat=["a", "a", "c"],),
    ).to_spark()
    return df


def test_create_model(sample_data_train, sample_data_holdout, sample_data_unseen):

    bc = BClassifier(
        training_data=sample_data_train,
        hold_out_data=sample_data_holdout,
        target_feature="label",
        numeric_features=["col_a", "col_b"],
        categorical_features=["col_cat"],
        session_id=345,
        impute_missing_values=False,
        missing_value_strategy="mean",
        robust_scale=True,
        min_max_scale=True,
        max_abs_scale=True,
        polynomial_feature=True,
        interaction=True,
    )

    # create model
    bc.create_model("lr")

    # make prediction
    pred = bc.predict_model(sample_data_unseen)

    # make sure that transformed data set has 'features' column
    assert "prediction" in pred.columns


def test_compare_model(sample_data_train, sample_data_holdout, sample_data_unseen):

    bc = BClassifier(
        training_data=sample_data_train,
        hold_out_data=sample_data_holdout,
        target_feature="label",
        numeric_features=["col_a",],
        categorical_features=["col_cat"],
        session_id=345,
        impute_missing_values=True,
        missing_value_strategy="mean",
        robust_scale=True,
        min_max_scale=True,
        max_abs_scale=True,
        polynomial_feature=True,
    )

    # create model
    bc.compare_models()

    # make prediction
    pred = bc.predict_model(sample_data_unseen)

    # make sure that transformed data set has 'features' column
    assert "prediction" in pred.columns

