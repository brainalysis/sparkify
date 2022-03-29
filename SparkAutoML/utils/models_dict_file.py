# make a dictionary of available models in pyspark

from pyspark.ml.classification import (
    LogisticRegression,
    GBTClassifier,
    RandomForestClassifier,
)

model_dict = {
    "lr": LogisticRegression,
    "rf": RandomForestClassifier,
    "gb": GBTClassifier,
}

