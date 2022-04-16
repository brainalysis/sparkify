""" this module calculate evaluation metrics for pyspark mlib """


from pyspark.sql import DataFrame as SparkDataFrame
import pyspark.pandas as ps
from pyspark.ml.evaluation import RegressionEvaluator


def evaluator(
    df: SparkDataFrame,
    model_name: str,
    target_column: str,
    data_set_type: str = "training",
):
    # r2
    r2_evaluator = RegressionEvaluator(
        labelCol=target_column, predictionCol="prediction", metricName="r2"
    )
    r2 = r2_evaluator.evaluate(df)

    # rmse
    rmse_evaluator = RegressionEvaluator(
        labelCol=target_column, predictionCol="prediction", metricName="rmse"
    )
    rmse = rmse_evaluator.evaluate(df)

    # mse
    mse_evaluator = RegressionEvaluator(
        labelCol=target_column, predictionCol="prediction", metricName="mse"
    )
    mse = mse_evaluator.evaluate(df)

    # mae
    mae_evaluator = RegressionEvaluator(
        labelCol=target_column, predictionCol="prediction", metricName="mae"
    )
    mae = mae_evaluator.evaluate(df)

    # result holder
    results = ps.DataFrame(
        dict(
            model=[model_name],
            type=[data_set_type],
            r2=[round(r2, 4)],
            rmse=[round(rmse, 4)],
            mse=[round(mse, 4)],
            mae=[round(mae, 4)],
        )
    )

    return results
