import os

from icecream import ic

from agroml.data import Data

def test_errorForNullFile():

    try:
        data = Data("nullFile.csv")
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
        data = Data(file)
        assert data._getFileExtension() == extension
    
def test_readDataCsv():
    allDataFiles = os.listdir("tests/testData")

    variableList = [
        "station","day","year","date","tx","tm","tn","rhx","rhm","rhn","rs","et0"
    ]

    for file in allDataFiles:
        data = Data("tests/testData/" + file)
        assert data.nColumns == len(variableList)
        ic(data.columnNamesList)
        ic(variableList)
        assert data.columnNamesList == variableList
        
def test_filterData():
    data = Data("tests/testData/dataExample.csv")
    dataFiltered = data.filterFeatures(variableList = ["tx", "tm", "tn"])
    assert dataFiltered.shape[1] == 3
    assert list(dataFiltered.columns) == ["tx", "tm", "tn"]
    assert dataFiltered.shape[0] == data.shape[0]
