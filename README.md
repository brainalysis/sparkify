# SparkAutoML
<img alt="PyPI" src="https://img.shields.io/pypi/v/SparkAutoML"> <img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/SparkAutoML"> <img alt="GitHub" src="https://img.shields.io/github/license/brainalysis/sparkify"> <img alt="GitHub language count" src="https://img.shields.io/github/languages/count/brainalysis/sparkify">[![Downloads](https://pepy.tech/badge/sparkautoml)](https://pepy.tech/project/sparkautoml)

#### ***This is a TRIAL version***

The main idea of the package is to easily build and deploy pyspark's machine learning library

If you want to contribute please reach out to fahadakbar@gmail.com

Thank you !


https://user-images.githubusercontent.com/19522276/163658399-17786b20-0208-44ff-b111-98b74ab34d25.mp4

---

## How to install
PySpark's version 3.2.1 or higher is required

```
pip install SparkAutoML
```
---


## How to use

**|Binary Classifier|** 

1: Import the classifier
```
from SparkAutoML.ml_module.BinaryClassifier_module.classifier_file import BClassifier
```

2: Setup the experimental
```
bcf = BClassifier(
    training_data=train, # spark data frame
    hold_out_data=test,  # spark data frame
    target_feature="target",  # target feature
    numeric_features=["num_1","num_3"], # list of all numeric features
    categorical_features=["cat_4", "cat_2"], # list of all categorical features
)
```

3: Create a model, e.g. random forest classifier
```
bcf.create_model("rfc") 
```
 **OR**

 Compare multiple models

```
bcf.compare_models(sort='auc')
```

4: Make predictions
```
bcf.predict_model(unseen_data # spark data frame)
```

5: Access the entire pipeline
```
bcf.fitted_pipeline
