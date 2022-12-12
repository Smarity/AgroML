
from agroml.data import Data, ModelData
from agroml.models import MultiLayerPerceptron


rawdata = Data("tests/testData/dataExample.csv")
modelData = ModelData(
    data = rawdata,
    inputList = ["tx", "tn", "rs"],
    outputList = ["et0"])
modelData.splitToTrainTest(splitFunction="SplitRandom", testSize=0.3)
modelData.normalizeData(method= "StandardScaler")


def test_initMultiLayerPerceptronModel():
    global modelData

    model = MultiLayerPerceptron(modelData)
    assert model.modelData == modelData



