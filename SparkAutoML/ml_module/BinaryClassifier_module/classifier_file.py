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
        print(f"Training successfully ended for {model_name} ....")

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

    def evaluate_model(self,evaluate_on:str="train") -> None:
        """evaluation on train or holdout data set"""
        # make final result holder
        self.evaluation_results = ps.DataFrame()

        # evaluate for training data set
        results = self.evaluator(
            self.train_data_transformed,
            self.model_name,
            self.target_feature,
            data_set_type=evaluate_on,
        )

        # append results
        if evaluate_on == "train":
            
            results = self.evaluator(
            self.train_data_transformed,
            self.model_name,
            self.target_feature,
            data_set_type=evaluate_on,)
            self.evaluation_results_training = results
        else:
            results = self.evaluator(
            self.hold_out_data_transformed,
            self.model_name,
            self.target_feature,
            data_set_type=evaluate_on,)
            self.evaluation_results_hold_out = results

        return results
    
    def compare_models(self,sort_by:str="auc"):
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
        results = (
            results.reset_index(drop=True)
            .sort_values(by=sort_by,ascending=False)
        )

        # keep the top performing pipeline
        self.fitted_pipeline = self.compare_model_dict[results.iloc[0,0]]
        self.evaluation_results_hold_out = results
            
        return results
        
        
        
