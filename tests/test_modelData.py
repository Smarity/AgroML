from agroml.data import Data, ModelData


def test_defineInputsAndOutputsFromInputData():
    inputData = importDataExampleCsv()
    modelData = ModelData(inputData)

def importDataExampleCsv():
    data = Data("tests/testData/dataExample.csv")
    return data

def test_defineVariablesNonInColumns():
    pass