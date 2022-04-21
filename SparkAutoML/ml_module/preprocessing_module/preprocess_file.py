from typing import List
from pyspark.sql import DataFrame as SparkDataFrame

from pyspark.ml.feature import (
    VectorAssembler,
    VectorIndexer,
    OneHotEncoder,
    StringIndexer,
    Imputer,
    Normalizer,
    StandardScaler,
    RobustScaler,
    MinMaxScaler,
    MaxAbsScaler,
    PolynomialExpansion,
)
from pyspark.ml import Pipeline
from pyspark.sql.types import DoubleType

from SparkAutoML.ml_module.preprocessing_module.custom_transformers_file import (
    Connector,
    ColumnHandler,
)


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
        session_id=123,
        impute_missing_values: bool = False,
        missing_value_strategy: str = "mean",
        normalize=False,
        normalize_p=2,
        standard_scale=False,
        standard_scaler_withMean=False,
        standard_scaler_withStd=True,
        robust_scale=False,
        robust_scale_withScaling=True,
        robust_scale_withCentering=False,
        robust_scale_lower=0.25,
        robust_scale_upper=0.75,
        min_max_scale=False,
        min_max_scale_min=0,
        min_max_scale_max=1,
        max_abs_scale=False,
        polynomial_feature=False,
        polynomial_degree=3,
    ) -> None:

        self.train_data = training_data
        self.hold_out_data = hold_out_data
        self.target_feature = target_feature
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.session_id = session_id
        self.impute_missing_values = impute_missing_values
        self.missing_value_strategy = missing_value_strategy
        self.normalize = normalize
        self.normalize_p = normalize_p
        self.standard_scale = standard_scale
        self.standard_scaler_withMean = standard_scaler_withMean
        self.standard_scaler_withStd = standard_scaler_withStd
        self.robust_scale = robust_scale
        self.robust_scale_withScaling = robust_scale_withScaling
        self.robust_scale_withCentering = robust_scale_withCentering
        self.robust_scale_lower = robust_scale_lower
        self.robust_scale_upper = robust_scale_upper
        self.min_max_scale = min_max_scale
        self.min_max_scale_min = min_max_scale_min
        self.min_max_scale_max = min_max_scale_max
        self.max_abs_scale = max_abs_scale
        self.polynomial_feature = polynomial_feature
        self.polynomial_degree = polynomial_degree

    # def _column_name_generator(self, input_cols: list, suffix: str) -> list:
    #     """This function adds suffix to the list of columns"""
    #     new_cols = [column + str(suffix) for column in input_cols]
    #     return new_cols

    def _cast_double_type(self, df: SparkDataFrame, column: str) -> SparkDataFrame:
        df = df.withColumn(column, df[column].cast(DoubleType()))
        return df

    # ====================== Builder ===================================

    def _builder(self,):
        # first we deploy imputer
        if self.impute_missing_values:
            imputer = Imputer(
                inputCols=self.numeric_features,
                outputCols=self.numeric_features,
                strategy=self.missing_value_strategy,
            )
        else:
            imputer = Connector()

        # we need Vector Assambler here anyway
        assemble_numeric = VectorAssembler(
            inputCols=self.numeric_features, outputCol="numeric_features"
        )

        # if we need normalizer
        if self.normalize:
            normalizer = Normalizer(
                inputCol="numeric_features", outputCol="numeric_features1"
            )
            column_handler1 = ColumnHandler(
                delete_col="numeric_features", replace_col="numeric_features1"
            )
        else:
            normalizer = Connector()
            column_handler1 = Connector()

        # standard scaler
        if self.standard_scale:
            standard_scaler = StandardScaler(
                inputCol="numeric_features",
                outputCol="numeric_features1",
                withMean=self.standard_scaler_withMean,
                withStd=self.standard_scaler_withStd,
            )
            column_handler2 = ColumnHandler(
                delete_col="numeric_features", replace_col="numeric_features1"
            )
        else:
            standard_scaler = Connector()
            column_handler2 = Connector()

        # robust Scaler
        if self.robust_scale:
            robust_scaler = RobustScaler(
                inputCol="numeric_features",
                outputCol="numeric_features1",
                withScaling=self.robust_scale_withScaling,
                withCentering=self.robust_scale_withCentering,
                lower=self.robust_scale_lower,
                upper=self.robust_scale_upper,
            )
            column_handler3 = ColumnHandler(
                delete_col="numeric_features", replace_col="numeric_features1"
            )
        else:
            robust_scaler = Connector()
            column_handler3 = Connector()

        # min max scaler
        if self.min_max_scale:
            min_max_scaler = MinMaxScaler(
                inputCol="numeric_features",
                outputCol="numeric_features1",
                min=self.min_max_scale_min,
                max=self.min_max_scale_max,
            )
            column_handler4 = ColumnHandler(
                delete_col="numeric_features", replace_col="numeric_features1"
            )
        else:
            min_max_scaler = Connector()
            column_handler4 = Connector()

        # Polynomial
        if self.polynomial_feature:
            polynomial = PolynomialExpansion(
                inputCol="numeric_features",
                outputCol="numeric_features1",
                degree=self.polynomial_degree,
            )
            column_handler6 = ColumnHandler(
                delete_col="numeric_features", replace_col="numeric_features1"
            )
        else:
            polynomial = Connector()
            column_handler6 = Connector()

        # Max Abs Scaler
        if self.max_abs_scale:
            max_abs_scaler = MaxAbsScaler(
                inputCol="numeric_features", outputCol="numeric_features1",
            )
            column_handler5 = ColumnHandler(
                delete_col="numeric_features", replace_col="numeric_features1"
            )
        else:
            max_abs_scaler = Connector()
            column_handler5 = Connector()

        # categorical_features
        # _________________________________________________________________________

        if self.categorical_features:
            str_indexer = StringIndexer(
                inputCols=self.categorical_features,
                outputCols=[
                    column + str("_indexer") for column in self.categorical_features
                ],
            )
            encoder = OneHotEncoder(
                inputCols=[
                    column + str("_indexer") for column in self.categorical_features
                ],
                outputCols=[
                    column + str("_encoder") for column in self.categorical_features
                ],
                dropLast=False,
            )

            onehot_assembler = VectorAssembler(
                inputCols=[
                    column + str("_encoder") for column in self.categorical_features
                ],
                outputCol="categorical_features",
            )

        else:
            str_indexer = Connector()
            encoder = Connector()
            onehot_assembler = Connector()

        # -------------Build Pipeline---------------
        self.pipeline = Pipeline(
            stages=[
                imputer,
                assemble_numeric,
                normalizer,
                column_handler1,
                standard_scaler,
                column_handler2,
                robust_scaler,
                column_handler3,
                min_max_scaler,
                column_handler4,
                max_abs_scaler,
                column_handler5,
                polynomial,
                column_handler6,
                str_indexer,
                encoder,
                onehot_assembler,
            ]
        )

        return None

    # ====================================Executions=================================================
    def run_pipeline(self):
        """ This is will run pipeline with some required steps"""

        # make sure target column is of double type
        self.train_data = self._cast_double_type(
            df=self.train_data, column=self.target_feature
        )
        self.hold_out_data = self._cast_double_type(
            df=self.hold_out_data, column=self.target_feature
        )
        # run pipeline Builder
        self._builder()

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

