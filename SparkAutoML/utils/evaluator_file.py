""" this module calculate evaluation metrics for pyspark mlib """

from pyspark.sql import DataFrame as SparkDataFrame
import pyspark.pandas as ps
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
)


def evaluator(
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
