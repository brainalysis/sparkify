""" This applies Binary Classification using models available in pyspark"""
from typing import List


from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
)
import pyspark.pandas as ps


from SparkAutoML.ml_module.preprocessing_module.preprocess_file import Preprocessor
from SparkAutoML.utils.models_dict_file import model_dict


class BClassifier(Preprocessor):
    def create_model(self, model_name: str,**args) -> None:
        print(f"Training started for {model_name} ....")
        # instantiate the model
        self.model_name = model_name
        self.model = model_dict[model_name]
        self.model = self.model(featuresCol="features", labelCol=self.target_feature,**args)

        # run the preprocessing pipeline
        self.run_pipeline()
        pipe = self.pipeline.getStages()
        pipe.append(self.model)
        self.pipeline.setStages(pipe)

        # now do fit transform
        self.fit_transform()
        # fit on hold out data set
        self.transform()
        print("Training successfully ended ....")

    def evaluator(
        self,
        df: SparkDataFrame,
        model_name: str,
        target_column: str,
        data_set_type: str = "training",
    ):
        # AUC
        bc = BinaryClassificationEvaluator(
            rawPredictionCol="rawPrediction",
            labelCol=target_column,
            metricName="areaUnderROC",
        )
        auc = bc.evaluate(df)
        # Accuracy
        mc = MulticlassClassificationEvaluator(
            labelCol="prediction", predictionCol=target_column
        )
        accuracy = mc.evaluate(df, {mc.metricName: "accuracy"})
        # precision
        precision = mc.evaluate(df, {mc.metricName: "weightedPrecision"})
        # recall
        recall = mc.evaluate(df, {mc.metricName: "weightedRecall"})
        # f1
        f1 = mc.evaluate(df, {mc.metricName: "f1"})

        results = ps.DataFrame(
            dict(
                model=[model_name],
                type=[data_set_type],
                auc=[auc],
                accuracy=[accuracy],
                weighted_precision=[precision],
                weighted_recall=[recall],
                f1=[f1],
            )
        )

        return results

    def evaluate_model(self) -> None:
        """evaluation on train and holdout data set"""
        # make final result holder
        self.evaluation_results = ps.DataFrame()

        # evaluate for training data set
        train_results = self.evaluator(
            self.train_data_transformed,
            self.model_name,
            self.target_feature,
            data_set_type="train",
        )

        # evaluate for test data set
        test_results = self.evaluator(
            self.hold_out_data_transformed,
            self.model_name,
            self.target_feature,
            data_set_type="hold_out",
        )

        # append results
        self.evaluation_results = self.evaluation_results.append(train_results)
        self.evaluation_results = self.evaluation_results.append(test_results)

        return self.evaluation_results

