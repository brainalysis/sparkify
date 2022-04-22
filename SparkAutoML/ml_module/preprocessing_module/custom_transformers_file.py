"""This module contains some custom pyspark transformers that will be used to fill the gaps in the pipeline"""
from pyspark import keyword_only
from pyspark.ml import Transformer
from pyspark.ml.param.shared import (
    HasInputCol,
    HasOutputCol,
    Param,
    Params,
    TypeConverters,
)

# Available in PySpark >= 2.3.0
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, StringType


class Connector(
    Transformer,
    HasInputCol,
    HasOutputCol,
    DefaultParamsReadable,
    DefaultParamsWritable,
):
    @keyword_only
    def __init__(self, inputCol=None, outputCol=None):
        """ This is used to just pass the data frame as is from the Pipeline, a filler and
        does nothing
        """
        super(Connector, self).__init__()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(
        self, inputCol=None, outputCol=None,
    ):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    # Required in Spark >= 3.0
    def setInputCol(self, value):
        """
        Sets the value of :py:attr:`inputCol`.
        """
        return self._set(inputCol=value)

    # Required in Spark >= 3.0
    def setOutputCol(self, value):
        """
        Sets the value of :py:attr:`outputCol`.
        """
        return self._set(outputCol=value)

    def _transform(self, dataset):
        dataset = dataset

        return dataset


class ColumnHandler(
    Transformer,
    HasInputCol,
    HasOutputCol,
    DefaultParamsReadable,
    DefaultParamsWritable,
):

    delete_col = Param(
        Params._dummy(),
        "delete_col",
        "delete_col",
        typeConverter=TypeConverters.toListString,
    )

    replace_col = Param(
        Params._dummy(),
        "replace_col",
        "replace_col",
        typeConverter=TypeConverters.toString,
    )

    @keyword_only
    def __init__(
        self, inputCol=None, outputCol=None, delete_col=None, replace_col=None
    ):
        super(ColumnHandler, self).__init__()
        self.delete_col = Param(self, "delete_col", "")
        self.replace_col = Param(self, "replace_col", "")
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(
        self, inputCol=None, outputCol=None, delete_col=None, replace_col=None
    ):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def setDeleteCol(self, value):
        return self._set(delete_col=value)

    def getDeleteCol(self):
        return self.getOrDefault(self.delete_col)

    def setReplaceCol(self, value):
        return self._set(replace_col=value)

    def getReplaceCol(self):
        return self.getOrDefault(self.replace_col)

    # Required in Spark >= 3.0
    def setInputCol(self, value):
        """
        Sets the value of :py:attr:`inputCol`.
        """
        return self._set(inputCol=value)

    # Required in Spark >= 3.0
    def setOutputCol(self, value):
        """
        Sets the value of :py:attr:`outputCol`.
        """
        return self._set(outputCol=value)

    def _transform(self, dataset):
        delete_col = self.getDeleteCol()
        replace_col = self.getReplaceCol()

        dataset = dataset.drop(*delete_col)
        dataset = dataset.withColumnRenamed(replace_col, delete_col[0])

        return dataset
