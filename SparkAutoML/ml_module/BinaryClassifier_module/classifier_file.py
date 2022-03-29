""" This applies Binary Classification using models available in pyspark"""
from typing import List

from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator


from SparkAutoML.ml_module.preprocessing_module.preprocess_file import Preprocessor
from SparkAutoML.utils.models_dict_file import model_dict


class BClassifier(Preprocessor):
    
    def create_model(self, model_name: str) -> None:
        # instantiate the model
        self.model = model_dict[model_name]
        self.model = self.model(featuresCol='features',labelCol=self.target_feature)
        
        # run the preprocessing pipeline
        self.run_pipeline()
        pipe = self.pipeline.getStages()
        pipe.append(self.model)
        self.pipeline.setStages(pipe)
        
        # now do fit transform
        self.fit_transform()
        # fit on hold out data set
        self.transform()
        
    def evaluator(self,df:SparkDataFrame,target_column):
        # AUC
        bc = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction",labelCol=target_column,metricName='areaUnderROC')
        auc= bc.evaluate(cf.train_data_transformed)
        # Accuracy
        mc = MulticlassClassificationEvaluator(labelCol='prediction',predictionCol=target_column)
        accuracy = mc.evaluate(cf.train_data_transformed, {mc.metricName:'accuracy'}) 
        
        
    
    def evaluate_model(self)-> None:
        """evaluation on train and holdout data set"""
        
        
        
        
        

