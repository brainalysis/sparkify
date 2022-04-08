from typing import List
from pyspark.sql import DataFrame as SparkDataFrame

from pyspark.ml.feature import (
    VectorAssembler,
    VectorIndexer,
    OneHotEncoder,
    StringIndexer,
    Imputer,
)
from pyspark.ml import Pipeline
from pyspark.sql.types import DoubleType


class Preprocessor:
    """
  Preprocess to prepare data for ML tasks
  """

    def __init__(
        self,
        training_data: SparkDataFrame,
        hold_out_data: SparkDataFrame,
        target_feature: "str",
        numeric_features: List[str],
        categorical_features: List[str],
        impute_missing_values: bool = False,
        missing_value_strategy: str = "mean",
    ) -> None:

        self.train_data = training_data
        self.hold_out_data = hold_out_data
        self.target_feature = target_feature
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.impute_missing_values = impute_missing_values
        self.missing_value_strategy = missing_value_strategy

    def _column_name_generator(self, input_cols: list, suffix: str) -> list:
        """This function adds suffix to the list of columns"""
        new_cols = [column + str(suffix) for column in input_cols]
        return new_cols

    def _cast_double_type(self, df: SparkDataFrame, column: str) -> SparkDataFrame:
        df = df.withColumn(column, df[column].cast(DoubleType()))
        return df

    # ====================== Transformers ===================================

    def _indexer(self, input_cols: list, suffix: str = "_index") -> None:
        """spark's indexer transformer"""
        self._indexed_cols = self._column_name_generator(input_cols, suffix)
        self.indexers = StringIndexer(
            inputCols=input_cols, outputCols=self._indexed_cols
        )
        return None

    def _encoder(self, input_cols: list, suffix: str = "_onehot") -> None:
        """spark's one hot encoder transformer"""
        self._encoded_cols = self._column_name_generator(input_cols, suffix)
        self.encoders = OneHotEncoder(
            inputCols=input_cols, outputCols=self._encoded_cols, dropLast=False
        )
        return None

    def _imputer(
        self, input_cols: list, suffix: str = "_imputed", strategy: str = "mean"
    ) -> None:
        self._imputed_cols = self._column_name_generator(input_cols, suffix)
        self.imputers = Imputer(
            inputCols=input_cols,
            outputCols=self._imputed_cols,
            strategy=self.missing_value_strategy,
        )
        return None

    def _vector_assembler(
        self, input_cols: list, output_cols: str = "features"
    ) -> None:
        """spark's Vector Assembler transformer"""
        self.vector = VectorAssembler(inputCols=input_cols, outputCol="features")
        return None

    def _pipeline(self, stages: list) -> None:
        """spark's Pipeline class """
        self.pipeline = Pipeline(stages=stages)
        return None

    # ===============================Pipelines==========================================
    def _pipeline_numeric(self) -> None:
        """->Vector Assembler"""
        self._vector_assembler(input_cols=self.numeric_features, output_cols="features")
        # pipeline
        self._pipeline([self.vector])
        return None

    def _pipeline_numeric_imputer(self) -> None:
        """->imputer -> Vector Assembler"""
        self._imputer(input_cols=self.numeric_features)
        self._vector_assembler(input_cols=self.numeric_features, output_cols="features")
        # pipeline
        self._pipeline([self.imputers, self.vector])

    def _pipeline_categorical(self) -> None:
        # transformers
        self._indexer(input_cols=self.categorical_features)
        self._encoder(input_cols=self._indexed_cols)
        self._vector_assembler(input_cols=self._encoded_cols, output_cols="features")
        # pipeline
        self._pipeline([self.indexers, self.encoders, self.vector])

    def _pipeline_mixed(self) -> None:
        """indexer -> Categorical-> Vector Assembler"""
        # transformers
        self._indexer(input_cols=self.categorical_features)
        self._encoder(input_cols=self._indexed_cols)
        self._vector_assembler(
            input_cols=self.numeric_features + self._encoded_cols,
            output_cols="features",
        )
        # pipeline
        self._pipeline([self.indexers, self.encoders, self.vector])
        return None

    def _pipeline_mixed_imputer(self) -> None:
        """indexer -> Categorical-> Vector Assembler"""
        # transformers
        self._indexer(input_cols=self.categorical_features)
        self._encoder(input_cols=self._indexed_cols)
        self._imputer(input_cols=self.numeric_features)
        self._vector_assembler(
            input_cols=self._imputed_cols + self._encoded_cols, output_cols="features"
        )
        # pipeline
        self._pipeline([self.indexers, self.encoders, self.imputers, self.vector])
        return None

    # ====================================Executions=================================================
    def run_pipeline(self):
        """ This is will run pipeline accoding to user selections"""

        # make sure target column is of double type
        self.train_data = self._cast_double_type(
            df=self.train_data, column=self.target_feature
        )
        self.hold_out_data = self._cast_double_type(
            df=self.hold_out_data, column=self.target_feature
        )

        # numeric only pipeline
        if (
            self.categorical_features is None
            and self.numeric_features is not None
            and self.impute_missing_values == False
        ):
            self._pipeline_numeric()

        # numeric + imputation
        if (
            self.categorical_features is None
            and self.numeric_features is not None
            and self.impute_missing_values == True
        ):
            self._pipeline_numeric_imputer()

        # categorical only
        if (
            self.categorical_features is not None
            and self.numeric_features is None
            and self.impute_missing_values == False
        ):
            self._pipeline_categorical()

        # mixed (numeric + categorical) pipeline
        if (
            self.categorical_features is not None
            and self.numeric_features is not None
            and self.impute_missing_values == False
        ):
            self._pipeline_mixed()

        # mixed + imputer  pipeline
        if (
            self.categorical_features is not None
            and self.numeric_features is not None
            and self.impute_missing_values == True
        ):
            self._pipeline_mixed_imputer()

        return None

    def fit_transform(self) -> None:
        """fit on the preprocessing file. Make sure to execute the run_pipeline method first"""
        self.fitted_pipeline = self.pipeline.fit(self.train_data)
        self.train_data_transformed = self.fitted_pipeline.transform(self.train_data)

        return None

    def transform(self) -> None:
        """transform hold out data"""
        self.hold_out_data_transformed = self.fitted_pipeline.transform(
            self.hold_out_data
        )

        return None

