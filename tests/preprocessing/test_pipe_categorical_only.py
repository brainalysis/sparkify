import pytest
import pyspark.pandas as ps

from SparkAutoML.ml_module.preprocessing_module.preprocess_file import Preprocessor


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
        dict(col_a=[5, 6, 8], col_b=[0.30, 0.80, 0.32,], col_cat=["a", "a", "c"],)
    ).to_spark()
    return df


def test_category_only(sample_data_train, sample_data_holdout, sample_data_unseen):

    pipe = Preprocessor(
        training_data=sample_data_train,
        hold_out_data=sample_data_holdout,
        target_feature="label",
        numeric_features=None,
        categorical_features=["col_cat"],
    )

    # execute the pipeline
    pipe.run_pipeline()
    pipe.fit_transform()
    pipe.transform()

    # make sure that transformed data set has 'features' column
    assert "features" in pipe.train_data_transformed.columns
    # for hold out
    assert "features" in pipe.hold_out_data_transformed.columns
    # for unseen
    assert "features" in pipe.fitted_pipeline.transform(sample_data_unseen).columns
