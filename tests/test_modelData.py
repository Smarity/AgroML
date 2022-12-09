import warnings

import pytest
import pandas as pd
from icecream import ic

from agroml.data import Data, ModelData

def test_filterInputsOutputs():
    data = Data("tests/testData/dataExample.csv")

    inputLists = [
        ["tx", "tm", "tn"],
        ["tx", "tm", "weirdFeature"]
    ]
    outputLists = [
        ["et0", "rs"],
        ["et0", "weirdFeature"]
    ]

    for inputList, outputList in zip(inputLists, outputLists):
        modelData = ModelData(data, inputList, outputList)

        assert modelData.inputData.shape[1] == len(modelData.inputList)
        assert modelData.outputData.shape[1] == len(modelData.outputList)
        assert modelData.inputData.shape[0] == modelData.outputData.shape[0]

@pytest.mark.xfail()
def test_noInputFeatureInData():
    data = Data("tests/testData/dataExample.csv")

    inputList = ["txx", "tmm", "tnn"] # no input exists
    outputList = ["et0", "rs"], # all exist

    modelData = ModelData(data, inputList, outputList)

@pytest.mark.xfail()
def test_noOutputFeatureInData():
    data = Data("tests/testData/dataExample.csv")

    inputList = ["tx", "tm", "tn"] # no input exists
    outputList = ["et000"]

    modelData = ModelData(data, inputList, outputList)

def test_splitToTrainTestRandom():
    data = Data("tests/testData/dataExample.csv")

    inputList = ["tx", "tm", "ra"]
    outputList = ["et0"]

    modelData = ModelData(data, inputList, outputList)

    for testSize in [0.1, 0.4]:
        modelData.splitToTrainTest(
            splitFunction = "SplitRandom",
            testSize = testSize,
            randomState = 42,
        )

        assert modelData.dataTrain.shape[0] + modelData.dataTest.shape[0] == modelData.data.shape[0]

def test_splitToTrainTestSequentially():
    data = Data("tests/testData/dataExample.csv")

    inputList = ["tx", "tm", "ra"]
    outputList = ["et0"]

    modelData = ModelData(data, inputList, outputList)

    modelData.splitToTrainTest(
        splitFunction = "SplitSequentially",
        testSize = 0.4,
    )

    assert modelData.dataTrain.shape[0] + modelData.dataTest.shape[0] == modelData.data.shape[0]

def test_splitToTrainTestByYear():
    pass

def test_splitToTrainTestByStation():
    pass

def test_normalizeDataReturnsPandasDAtaFrame():
    data = Data("tests/testData/dataExample.csv")

    inputList = ["tx", "tm", "ra"]
    outputList = ["et0"]

    modelData = ModelData(data, inputList, outputList)

    for method in ["StandardScaler", "MinMaxScaler"]:
        modelData.normalizeData(method = method)

        assert isinstance(modelData._xTrain, pd.DataFrame)
        assert isinstance(modelData.xTrain, pd.DataFrame)

        assert isinstance(modelData._xTest, pd.DataFrame)
        assert isinstance(modelData.xTest, pd.DataFrame)



    