import pytest
import pyspark.pandas as ps
from pyspark.sql import SparkSession

from SparkAutoML.ml_module.preprocessing_module.preprocess_file import Preprocessor

spark = SparkSession.builder.appName("test_pipeline").getOrCreate()

@pytest.fixture
def sample_data_train():
    # df = spark.createDataFrame([
    #   (1,1,.24),
    #   (0,2,.56),
    #   (0,3,.12),
    #   (1,4,.39)
    # ],
    #   ['col_a','col_b','label'])
    df = ps.DataFrame(dict(col_a=[1,2,3,4],col_b=[.24,.56,.12,.39],label=[1,0,0,1])).to_spark()
    return df

@pytest.fixture
def sample_data_holdout():
      # df = spark.createDataFrame([
      #  (1,1,.24),
      #  (0,2,.56),
      #  (0,3,.12),
      #  (1,4,.39)
      #   ],
      # ['col_a','col_b','label'])
    
    df = ps.DataFrame(dict(col_a=[5,6,,8],col_b=[.30,.80,.32,],label=[1,0,0,])).to_spark()
    return df
                                                                              
def test_numeric_only(sample_data_train ,sample_data_holdout):
                                                                              
    pipe = Preprocessor(training_data=sample_data_train,
                     hold_out_data=sample_data_holdout,
                     target_feature='label',
                     numeric_features= ['col_a','col_b'],
                    categorical_features= None)
    
    #execute the pipelne
    pipe.run_pipeline()
    pipe.fit_transform()
    pipe.transform()
                                                                              
    # make sure that transformed data set has 'features' column
    assert "features" in pipe.train_data_transformed.columns
    # for hold out
    assert "features" in pipe.hold_out_data_transformed.columns
                                                                            
                                                                              
                                                                              
                                                                              
    
