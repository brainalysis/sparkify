from typing import List
import pyspark.pandas as ps
from pyspark.sql import DataFrame as SparkDataFrame

from pyspark.ml.feature import VectorAssembler,VectorIndexer,OneHotEncoder,StringIndexer
from pyspark.ml import Pipeline 

from SparkAutoML.ml_module.BinaryClassifier_module.classifier_file import BClassifier

# read data 
df = ps.read_csv("credit.csv")

# that is how you can apply the full stack piping in koalas
df= (
df
[['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE','default']]
.pipe(lambda x: ps.sql(""" select * , case when LIMIT_BAL < 100000 then 'low' when LIMIT_BAL between 100000 and 200000 then 'medium' else 'high' end as BAL_CAT from {x}"""))
.pipe(lambda x: ps.sql(""" select * , case when AGE < 30 then 'youth' when AGE between 30 and 40 then 'yong' else 'mature' end as maturity from {x}"""))
)

spark_df = df.to_spark()
train,test = spark_df.randomSplit([.70,.30],seed=123)



cf = BClassifier(training_data=train
                     ,hold_out_data=test,
                     target_feature='default',
                     numeric_features= ['LIMIT_BAL',],
                     categorical_features=["BAL_CAT","maturity"],
                     impute_missing_values=True, missing_value_strategy="mean",
                    )

cf.create_model('dt')

print("done")