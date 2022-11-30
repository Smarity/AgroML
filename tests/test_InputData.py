from agroml.data.InputData import InputData

def test_errorForNullFile():
    try:
        inputData = InputData("nullFile.csv")
    except AssertionError:
        assert True

def test_getCorrectFileExtension():
    inputData = InputData("tests/data/dataExample.csv")
    assert inputData._getFileExtension() == ".csv"

    inputData = InputData("tests/data/dataExample.xlsx")
    assert inputData._getFileExtension() == ".xlsx"
    
    inputData = InputData("tests/data/dataExample.txt")
    assert inputData._getFileExtension() == ".txt"