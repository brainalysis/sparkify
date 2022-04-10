# make a dictionary of available models in pyspark

from pyspark.ml.classification import (
    LogisticRegression,
    GBTClassifier,
    RandomForestClassifier,
    DecisionTreeClassifier,
    MultilayerPerceptronClassifier,
    LinearSVC,
    NaiveBayes,
    FMClassifier,
)

model_dict = {
    "lrc": LogisticRegression,
    "rfc": RandomForestClassifier,
    "gbc": GBTClassifier,
    "dtc": DecisionTreeClassifier,
    "mlpc": MultilayerPerceptronClassifier,
    "svc": LinearSVC,
    "nbc": NaiveBayes,
    "fmc": FMClassifier,
}

