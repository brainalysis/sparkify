from typing import List
from pyspark.sql import DataFrame as SparkDataFrame

from pyspark.ml.feature import (
    VectorAssembler,
    OneHotEncoder,
    StringIndexer,
    Imputer,
    Normalizer,
    StandardScaler,
    RobustScaler,
    MinMaxScaler,
    MaxAbsScaler,
    PolynomialExpansion,
    Interaction,
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
        interaction=False,
        features_to_interact=None,
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
        self.interaction = interaction
        self.features_to_interact = features_to_interact

    # def _column_name_generator(self, input_cols: list, suffix: str) -> list:
    #     """This function adds suffix to the list of columns"""
    #     new_cols = [column + str(suffix) for column in input_cols]
    #     return new_cols

    def _cast_double_type(self, df: SparkDataFrame, column: str) -> SparkDataFrame:
        df = df.withColumn(column, df[column].cast(DoubleType()))
        return df

    # input selector for interaction feature
    def _input_for_interaction(self,):
        # if user does not give a list of features to interact
        # we take all available features
        if not self.features_to_interact:
            self.features_to_interact = []
            if self.numeric_features:
                # if we have numeric columns
                self.features_to_interact = (
                    self.features_to_interact + self.numeric_features
                )
            if self.categorical_features:
                # if we only have categorical
                self.features_to_interact = (
                    self.features_to_interact + self.categorical_features
                )

        # start making input columns
        if not self.numeric_features:
            # if we have only categorical columns
            input_cols = self.encoder_cols
            # if we have only numeric features
        elif not self.categorical_features:
            input_cols = self.numeric_features
        # if we have both type of features
        else:
            input_cols = [
                column + str("_encoder")
                if column in self.categorical_features
                else column
                for column in self.features_to_interact
            ]

        return input_cols

    # function to put all individual assemblers to one assembler
    def _final_features_list(self) -> list:

        # check if numeric, categorical and interaction columns
        # are required, if so , make a list of them

        # make a lis that has the details
        _list = [
            ("numeric_features", self.numeric_features),
            ("categorical_features", self.categorical_features),
            ("interaction_features", self.interaction),
        ]

        # add to list if selects were true
        result = [i[0] for i in _list if i[1]]

        return result

    # ====================== Builder ===================================

    def _builder(self,):
        # first we deploy imputer

        if self.impute_missing_values and self.numeric_features:
            imputer = Imputer(
                inputCols=self.numeric_features,
                outputCols=self.numeric_features,
                strategy=self.missing_value_strategy,
            )
        else:
            imputer = Connector()

        # we need Vector Assambler here for numeric features
        if self.numeric_features:
            assemble_numeric = VectorAssembler(
                inputCols=self.numeric_features, outputCol="numeric_features"
            )
        else:
            assemble_numeric = Connector()

        # if we need normalizer
        if self.normalize and self.numeric_features:
            normalizer = Normalizer(
                inputCol="numeric_features", outputCol="numeric_features1"
            )
            column_handler1 = ColumnHandler(
                delete_col=["numeric_features"], replace_col="numeric_features1"
            )
        else:
            normalizer = Connector()
            column_handler1 = Connector()

        # standard scaler
        if self.standard_scale and self.numeric_features:
            standard_scaler = StandardScaler(
                inputCol="numeric_features",
                outputCol="numeric_features1",
                withMean=self.standard_scaler_withMean,
                withStd=self.standard_scaler_withStd,
            )
            column_handler2 = ColumnHandler(
                delete_col=["numeric_features"], replace_col="numeric_features1"
            )
        else:
            standard_scaler = Connector()
            column_handler2 = Connector()

        # robust Scaler
        if self.robust_scale and self.numeric_features:
            robust_scaler = RobustScaler(
                inputCol="numeric_features",
                outputCol="numeric_features1",
                withScaling=self.robust_scale_withScaling,
                withCentering=self.robust_scale_withCentering,
                lower=self.robust_scale_lower,
                upper=self.robust_scale_upper,
            )
            column_handler3 = ColumnHandler(
                delete_col=["numeric_features"], replace_col="numeric_features1"
            )
        else:
            robust_scaler = Connector()
            column_handler3 = Connector()

        # min max scaler
        if self.min_max_scale and self.numeric_features:
            min_max_scaler = MinMaxScaler(
                inputCol="numeric_features",
                outputCol="numeric_features1",
                min=self.min_max_scale_min,
                max=self.min_max_scale_max,
            )
            column_handler4 = ColumnHandler(
                delete_col=["numeric_features"], replace_col="numeric_features1"
            )
        else:
            min_max_scaler = Connector()
            column_handler4 = Connector()

        # Max Abs Scaler
        if self.max_abs_scale and self.numeric_features:
            max_abs_scaler = MaxAbsScaler(
                inputCol="numeric_features", outputCol="numeric_features1",
            )
            column_handler5 = ColumnHandler(
                delete_col=["numeric_features"], replace_col="numeric_features1"
            )
        else:
            max_abs_scaler = Connector()
            column_handler5 = Connector()

        # Polynomial
        if self.polynomial_feature and self.numeric_features:
            polynomial = PolynomialExpansion(
                inputCol="numeric_features",
                outputCol="numeric_features1",
                degree=self.polynomial_degree,
            )
            column_handler6 = ColumnHandler(
                delete_col=["numeric_features"], replace_col="numeric_features1"
            )
        else:
            polynomial = Connector()
            column_handler6 = Connector()

        # categorical_features
        # _________________________________________________________________________

        if self.categorical_features:

            # column names for string indexer
            index_cols = [
                column + str("_indexer") for column in self.categorical_features
            ]
            # column name for one hot encoder
            self.encoder_cols = [
                column + str("_encoder") for column in self.categorical_features
            ]
            # start making blocks
            str_indexer = StringIndexer(
                inputCols=self.categorical_features, outputCols=index_cols,
            )
            encoder = OneHotEncoder(
                inputCols=index_cols, outputCols=self.encoder_cols, dropLast=False,
            )

            onehot_assembler = VectorAssembler(
                inputCols=self.encoder_cols, outputCol="categorical_features",
            )

            delete_indexer_columns = ColumnHandler(
                delete_col=index_cols, replace_col=None,
            )

        else:
            str_indexer = Connector()
            encoder = Connector()
            onehot_assembler = Connector()
            delete_indexer_columns = Connector()

        # interaction features
        # _________________________________________________________________________

        if self.interaction:

            # determine in put columns
            inter_input_cols = self._input_for_interaction()
            interactor = Interaction(
                inputCols=inter_input_cols, outputCol="interaction_features"
            )

        else:
            interactor = Connector()

        # Final Vector Assembler
        # ---------------------------------------------------------------------------

        # put together all the features through vector assembler
        input_cols_final = self._final_features_list()

        final_assembler = VectorAssembler(
            inputCols=input_cols_final, outputCol="features"
        )

        # remove unwanted columns
        delete_unwanted_columns_final = ColumnHandler(
            delete_col=self.encoder_cols + input_cols_final
            if self.categorical_features
            else input_cols_final,
            replace_col=None,
        )

        # ==========================================================================
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
                delete_indexer_columns,
                interactor,
                final_assembler,
                delete_unwanted_columns_final,
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

