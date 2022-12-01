import warnings

import pytest
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
    outputList = ["et000"], # all exist

    modelData = ModelData(data, inputList, outputList)


