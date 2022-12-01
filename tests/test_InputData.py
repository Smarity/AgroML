from agroml.data.InputData import InputData

from icecream import icecream

def test_errorForNullFile():
    try:
        inputData = InputData("nullFile.csv")
    except Exception as error:
        assert True
    
def test_getCorrectFileExtension():
    
    
    inputData = InputData("tests/testData/dataExample.csv")
    assert inputData._getFileExtension() == ".csv"

    inputData = InputData("tests/testData/dataExample.xlsx")
    assert inputData._getFileExtension() == ".xlsx"
    
    inputData = InputData("tests/testData/dataExample.txt")
    assert inputData._getFileExtension() == ".txt"
