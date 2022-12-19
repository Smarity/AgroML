from icecream import ic

from agroml.data import Data, ModelData
from agroml.cashOptimization import BayesianOptimization


rawdata = Data("tests/testData/dataExample.csv")
modelData = ModelData(
    data = rawdata,
    inputList = ["tx", "tn", "rs"],
    outputList = ["et0"])
modelData.splitToTrainTest(splitFunction="SplitRandom", testSize=0.3)
modelData.normalizeData(method= "StandardScaler")

def test_initialization():
    global modelData

    bayesianOptimization = BayesianOptimization(modelData)

    assert bayesianOptimization.modelData == modelData
    assert bayesianOptimization.splitFunction == "None"
    assert bayesianOptimization.validationSize == 0.2
    assert bayesianOptimization.randomSeed == 42
    assert bayesianOptimization.nFolds == 5
    assert bayesianOptimization.xVal is None
    assert bayesianOptimization.yVal is None

def test_splitToValidationUsingRandom():
    global modelData

    bayesianOptimization = BayesianOptimization(
        modelData = modelData,
        splitFunction = "SplitRandom",
        validationSize = 0.3,
        )

    assert len(bayesianOptimization.xTrain.shape) == 3
    assert len(bayesianOptimization.xVal.shape) == 3
    assert len(bayesianOptimization.yTrain.shape) == 3
    assert len(bayesianOptimization.yVal.shape) == 3

    assert bayesianOptimization.xTrain.shape[0] == modelData.xTrain.shape[0] - bayesianOptimization.xVal.shape[0]
    assert modelData.xTrain.shape[0] == bayesianOptimization.xTrain.shape[0] + bayesianOptimization.xVal.shape[0]

    assert bayesianOptimization.yTrain.shape[0] == modelData.yTrain.shape[0] - bayesianOptimization.yVal.shape[0]
    assert modelData.yTrain.shape[0] == bayesianOptimization.yTrain.shape[0] + bayesianOptimization.yVal.shape[0]

    assert bayesianOptimization.xTrainIndex.is_monotonic_increasing
    assert bayesianOptimization.xValIndex.is_monotonic_increasing

    assert bayesianOptimization.yTrainIndex.is_monotonic_increasing
    assert bayesianOptimization.yValIndex.is_monotonic_increasing

def test_splitToValidationUsingSequential():
    global modelData

    bayesianOptimization = BayesianOptimization(
        modelData = modelData,
        splitFunction = "SplitSequential",
        validationSize = 0.3,
        )

    assert len(bayesianOptimization.xTrain.shape) == 3
    assert len(bayesianOptimization.xVal.shape) == 3
    assert len(bayesianOptimization.yTrain.shape) == 3
    assert len(bayesianOptimization.yVal.shape) == 3

    assert bayesianOptimization.xTrain.shape[0] == modelData.xTrain.shape[0] - bayesianOptimization.xVal.shape[0]
    assert modelData.xTrain.shape[0] == bayesianOptimization.xTrain.shape[0] + bayesianOptimization.xVal.shape[0]

    assert bayesianOptimization.yTrain.shape[0] == modelData.yTrain.shape[0] - bayesianOptimization.yVal.shape[0]
    assert modelData.yTrain.shape[0] == bayesianOptimization.yTrain.shape[0] + bayesianOptimization.yVal.shape[0]

    assert bayesianOptimization.xTrainIndex.is_monotonic_increasing
    assert bayesianOptimization.xValIndex.is_monotonic_increasing

    assert bayesianOptimization.yTrainIndex.is_monotonic_increasing
    assert bayesianOptimization.yValIndex.is_monotonic_increasing


def test_splitToValidationUsingKFold():
    pass