from agroml.data import InputData, ModelData


def test_defineInputsAndOutputsFromInputData():
    inputData = importDataExampleCsv()
    modelData = ModelData(inputData)

def importDataExampleCsv():
    inputData = InputData("tests/testData/dataExample.csv")
    return inputData

def test_defineVariablesNonInColumns():
    pass