import os

import pytest
import pandas as pd
from numpy.testing import assert_allclose
from icecream import ic

from agroml.preprocessing import StandardScaler, MinMaxScaler

dfData = pd.read_csv("tests/testData/dataExample.csv", sep=";")

lenTrain = int(0.8*len(dfData))
xTrain = dfData.filter(items=["tx", "tm", "rs"]).iloc[:lenTrain]
xTest = dfData.filter(items=["tx", "tm", "rs"]).iloc[lenTrain:]
yTrain = dfData.filter(items=["et0"]).iloc[:lenTrain]
yTest = dfData.filter(items=["et0"]).iloc[lenTrain:]

def test_normalizeDataReturnsPandasDAtaFrame():
    global xTrain, xTest, yTrain, yTest

    for scaler in [StandardScaler, MinMaxScaler]:
        scaler = scaler(xTrain)
        xTrainScaled = scaler.transform(xTrain)
        xTestScaled = scaler.transform(xTest)

        assert isinstance(xTrainScaled, pd.DataFrame)
        assert isinstance(xTestScaled, pd.DataFrame)

def test_standardScalerReturnDataWithNoMeanNoStd():
    global xTrain, xTest, yTrain, yTest

    scaler = StandardScaler(xTrain)
    xTrainScaled = scaler.transform(xTrain)
    for col in xTrain.columns:
        assert_allclose(xTrainScaled[col].mean(), 0, atol=1e-03)
        assert_allclose(xTrainScaled[col].std(), 1, atol=1e-03)

def test_savingScaler():
    global xTrain, xTest, yTrain, yTest

    path = "tests/testScaler/scaler.pkl"
    if os.path.exists(path):
        os.remove(path)

    scaler = StandardScaler(xTrain)
    scaler.save("tests/testScaler/scaler")

    assert os.path.exists("tests/testScaler/scaler.pkl")

def test_loadingScaler():
    global xTrain, xTest, yTrain, yTest

    path = "tests/testScaler/scaler.pkl"
    try:
        scaler = StandardScaler(xTrain, path = path)
        xTrainScaled = scaler.transform(xTrain)
        assert True
    except:
        assert False

@pytest.mark.filterwarnings("ignore: The scaler does not exist")
def test_loadingNonExistentScaler():
    global xTrain, xTest, yTrain, yTest

    path = "tests/testScaler/scaler2.pkl"
    try:
        scaler = StandardScaler(xTrain, path = path)
    except:
        assert True

# Not implemented yet
def test_loadingTheIncorrectScaler():
    pass

def test_getParams():
    global xTrain, xTest, yTrain, yTest

    scaler = StandardScaler(xTrain)
    scaler.save("tests/testScaler/scaler")

    for scaler in [StandardScaler, MinMaxScaler]:
        scaler = scaler(xTrain)
        ic(scaler.getParams())
        assert scaler.getParams() == scaler.__str__()