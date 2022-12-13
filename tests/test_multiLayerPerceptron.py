
from tensorflow.keras import Model as TensorflowModel

from agroml.data import Data, ModelData
from agroml.models.regressionModels import MultiLayerPerceptron


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
    assert model.featuresList == modelData.inputList
    assert model.nFeatures == len(modelData.inputList)
    assert model.outputsList == modelData.outputList
    assert model.nOutputs == len(modelData.outputList)
    assert model.xTrain.shape[1] == model.nFeatures

def test_buildModel():
    global modelData

    model = MultiLayerPerceptron(modelData)
    model.buildModel()
    assert isinstance(model.model, TensorflowModel)

