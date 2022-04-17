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

from pyspark.ml.regression import (
    LinearRegression,
    GeneralizedLinearRegression,
    DecisionTreeRegressor,
    RandomForestRegressor,
    GBTRegressor,
    AFTSurvivalRegression,
    IsotonicRegression,
    FMRegressor,
)

model_dict_classifier = {
    "lr": LogisticRegression,
    "rfc": RandomForestClassifier,
    "gbc": GBTClassifier,
    "dtc": DecisionTreeClassifier,
    "mlpc": MultilayerPerceptronClassifier,
    "svc": LinearSVC,
    "nbc": NaiveBayes,
    "fmc": FMClassifier,
}


model_dict_regression = {
    "lr": LinearRegression,
    "glr": GeneralizedLinearRegression,
    "dtr": DecisionTreeRegressor,
    "rfr": RandomForestRegressor,
    "gbr": GBTRegressor,
    "sr": AFTSurvivalRegression,
    "isor": IsotonicRegression,
    "fmr": FMRegressor,
}
