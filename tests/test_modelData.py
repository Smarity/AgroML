import warnings
import os

import pytest
import pandas as pd
from icecream import ic

from agroml.data import Data, ModelData

def test_filterInputsOutputs():
    data = Data("tests/testData/dataExample.csv")

    inputLists = [
        ["tx", "tm", "tn"],
        ["tx", "tm"]
    ]
    outputLists = [
        ["et0", "rs"],
        ["et0"]
    ]

    for inputList, outputList in zip(inputLists, outputLists):
        modelData = ModelData(data, inputList, outputList)

        assert modelData.inputData.shape[1] == len(modelData.inputList)
        assert modelData.outputData.shape[1] == len(modelData.outputList)
        assert modelData.inputData.shape[0] == modelData.outputData.shape[0]

def test_noInputFeatureInData():
    data = Data("tests/testData/dataExample.csv")

    inputList = ["txx", "tmm", "tnn"] # no input exists
    outputList = ["et0", "rs"] # all exist

    with pytest.warns(UserWarning):
        modelData = ModelData(data, inputList, outputList)
        warnings.warn("Some features*", UserWarning)

def test_noOutputFeatureInData():
    data = Data("tests/testData/dataExample.csv")

    inputList = ["tx", "tm", "tn"] # no input exists
    outputList = ["et000"]

    with pytest.warns(UserWarning):
        modelData = ModelData(data, inputList, outputList)
        warnings.warn("Some features*", UserWarning)

def test_splitToTrainTestRandom():
    data = Data("tests/testData/dataExample.csv")

    inputList = ["tx", "tm", "rs"]
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

    inputList = ["tx", "tm", "rs"]
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

def test_normalizeDataReturnsPandasDataFrame():
    data = Data("tests/testData/dataExample.csv")

    inputList = ["tx", "tm", "rs"]
    outputList = ["et0"]

    modelData = ModelData(data, inputList, outputList)
    modelData.splitToTrainTest() # It splist using the random function

    for method in ["StandardScaler", "MinMaxScaler"]:
        modelData.normalizeData(method = method)

        assert isinstance(modelData._xTrain, pd.DataFrame)
        assert isinstance(modelData.xTrain, pd.DataFrame)

        assert isinstance(modelData._xTest, pd.DataFrame)
        assert isinstance(modelData.xTest, pd.DataFrame)

def test_saveScaler():
    data = Data("tests/testData/dataExample.csv")

    inputList = ["tx", "tm", "rs"]
    outputList = ["et0"]

    modelData = ModelData(data, inputList, outputList)
    modelData.splitToTrainTest() # It splist using the random function

    path = "tests/testScaler/scalerModelDataTest.pkl"
    if os.path.exists(path):
        os.remove(path)

    for method in ["StandardScaler", "MinMaxScaler"]:
        modelData.normalizeData(method = method)
        modelData.saveScaler(path="tests/testScaler/scalerModelDataTest.pkl")

        assert modelData._scaler is not None
        assert modelData._scalerPath == "tests/testScaler/scalerModelDataTest.pkl"
        assert os.path.exists(path)

        if os.path.exists(path): # remove again
            os.remove(path)

# I don't know how to test this
def test_avoidTwoNormalizations():
    pass

def test_tryToSaveScalerWithoutNormalization():
    data = Data("tests/testData/dataExample.csv")

    inputList = ["tx", "tm", "rs"]
    outputList = ["et0"]

    modelData = ModelData(data, inputList, outputList)
    modelData.splitToTrainTest() # It splist using the random function

    for method in ["StandardScaler", "MinMaxScaler"]:
        with pytest.warns(UserWarning):
            modelData.saveScaler(path="tests/testScaler/scalerModelDataTest.pkl")
            warnings.warn("You have to*", UserWarning)
    