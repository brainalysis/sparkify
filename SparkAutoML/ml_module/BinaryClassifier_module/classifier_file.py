""" This applies Binary Classification using models available in pyspark"""
from typing import List


from pyspark.sql import DataFrame as SparkDataFrame
import pyspark.pandas as ps


from SparkAutoML.ml_module.preprocessing_module.preprocess_file import Preprocessor
from SparkAutoML.utils.models_dict_file import model_dict
from SparkAutoML.utils.BClassifier_evaluator_file import evaluator


class BClassifier(Preprocessor):
    def create_model(self, model_name: str, **args) -> None:
        print(f"Training started for {model_name} ....")
        # instantiate the model
        self.model_name = model_name
        self.model = model_dict[model_name]
        self.model = self.model(
            featuresCol="features", labelCol=self.target_feature, **args
        )

        # run the preprocessing pipeline
        self.run_pipeline()
        pipe = self.pipeline.getStages()
        pipe.append(self.model)
        self.pipeline.setStages(pipe)

        # now do fit transform
        self.fit_transform()
        # fit on hold out data set
        self.transform()
        print(f"Training successfully ended for {model_name} ....")

    def evaluate_model(self, evaluate_on: str = "train") -> None:
        """evaluation on train or holdout data set"""
        if evaluate_on == "train":

            results = evaluator(
                self.train_data_transformed,
                self.model_name,
                self.target_feature,
                data_set_type=evaluate_on,
            )
            self.evaluation_results_training = results
        else:
            results = evaluator(
                self.hold_out_data_transformed,
                self.model_name,
                self.target_feature,
                data_set_type=evaluate_on,
            )
            self.evaluation_results_hold_out = results

        return results

    def compare_models(self, sort_by: str = "auc"):
        """use available models in mlib and evaluate them on holdout dataset"""

        results = ps.DataFrame()
        self.compare_model_dict = {}
        # create a placeholder
        for model in model_dict.keys():
            # try to train all models
            try:
                self.create_model(model)
                eval = self.evaluate_model(evaluate_on="holdout")
                results = results.append(eval)
                self.compare_model_dict[model] = self.fitted_pipeline
            except:
                pass

        # sort results by user choice
        results = results.reset_index(drop=True).sort_values(
            by=sort_by, ascending=False
        )

        # keep the top performing pipeline
        self.fitted_pipeline = self.compare_model_dict[results.iloc[0, 0]]
        self.evaluation_results_hold_out = results

        return results

    def predict_model(self, df: SparkDataFrame) -> SparkDataFrame:
        """make prediction on future dataset (data without label) """

        # keep the top performing pipeline
        results = self.fitted_pipeline.transform(df)

        return results
