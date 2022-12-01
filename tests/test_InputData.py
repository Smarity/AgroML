import os

from icecream import ic

from agroml.data import InputData

def test_errorForNullFile():

    try:
        inputData = InputData("nullFile.csv")
    except Exception as error:
        assert True
    
def test_getCorrectFileExtension():

    inputFilesList = [
        "tests/testData/dataExample.csv",
        "tests/testData/dataExample.xls",
        "tests/testData/dataExample.txt"
    ]
    fileExtensionList = [
        ".csv", ".xls", ".txt"
    ]
    for file, extension in zip(inputFilesList, fileExtensionList):
        inputData = InputData(file)
        assert inputData._getFileExtension() == extension
    
def test_readDataCsv():
    allDataFiles = os.listdir("tests/testData")

    variableList = [
        "station","day","year","date","tx","tm","tn","rhx","rhm","rhn","rs","et0"
    ]

    for file in allDataFiles:
        inputData = InputData("tests/testData/" + file)
        assert inputData.nColumns == len(variableList)
        ic(inputData.columnNamesList)
        ic(variableList)
        assert inputData.columnNamesList == variableList