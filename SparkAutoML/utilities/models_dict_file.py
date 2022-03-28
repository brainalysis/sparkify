# make a dictionary of available models in pyspark

from pyspark.ml.classification import LogisticRegression

model_dict = {"lr": LogisticRegression}

