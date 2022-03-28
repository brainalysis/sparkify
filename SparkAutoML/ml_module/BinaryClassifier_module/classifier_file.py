""" This applies Binary Classification using models available in pyspark"""
from typing import List

from pyspark.ml import Pipeline
from pyspark.sql import DataFrame as SparkDataFrame


from SparkAutoML.ml_module.preprocessing_module.preprocess_file import Preprocessor
from SparkAutoML.utilities.models_dict_file import model_dict


class BClassifier(Preprocessor):
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

        super().__init__(
            training_data,
            hold_out_data,
            target_feature,
            numeric_features,
            categorical_features,
            impute_missing_values,
            missing_value_strategy,
        )

        return None

    def create_model(self, model_name: str) -> None:
        # first run the preprocessing pipeline
        self.run_pipeline()
        pipe = self.pipeline.getStages()
        pipe.append(model_dict[model_name])
        self.pipeline.setStages(pipe)
        
        

