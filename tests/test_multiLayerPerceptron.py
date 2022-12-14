import os

import pytest

from tensorflow.keras import Model as TensorflowModel
from icecream import ic

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

    model.buildModel(
        nHiddenLayers=2,
        neuronsPerLayerList=[10,10],
        activation="relu",
        optimizer="Adam"
    )
    assert isinstance(model.model, TensorflowModel)

def test_canTwoModelsBeCompared():
    global modelData

    model1 = MultiLayerPerceptron(modelData)
    model1.buildModel()
    model2 = MultiLayerPerceptron(modelData)
    model2.buildModel()

    assert model1 == model2

    model2.buildModel(activation="sigmoid")
    assert model1 != model2


def test_buildModelThatFails():
    global modelData

    model = MultiLayerPerceptron(modelData)

    with pytest.raises(ValueError):
        model.buildModel(
            nHiddenLayers=2,
            neuronsPerLayerList=[10],
            activation="relu",
            optimizer="Adam"
        )

def test_buildModelThatWarns():
    global modelData

    model = MultiLayerPerceptron(modelData)

    with pytest.warns(UserWarning):
        model.buildModel(
            nHiddenLayers=1,
            neuronsPerLayerList=[10, 20],
            activation="relu",
            optimizer="Adam"
        )
    
    with pytest.warns(UserWarning):
        model.buildModel(
            nHiddenLayers=1,
            neuronsPerLayerList=[2],
            activation="reul",
            optimizer="Adam"
        )
        assert model.activation == "relu"
    
    with pytest.warns(UserWarning):
        model.buildModel(
            nHiddenLayers=1,
            neuronsPerLayerList=[2],
            activation="relu",
            optimizer="Adm"
        )
        assert model.optimizer == "Adam"
    
    with pytest.warns(UserWarning):
        model.buildModel(
            nHiddenLayers=1,
            neuronsPerLayerList=[2],
            activation="reu",
            optimizer="Adm"
        )
        assert model.activation == "relu"
        assert model.optimizer == "Adam"

def test_saveModel():
    global modelData

    model = MultiLayerPerceptron(modelData)
    model.buildModel()

    if os.path.isfile("tests/testModel/ModelMLP.h5"):
        os.remove("tests/testModel/ModelMLP.h5")

    model.saveModel("tests/testModel/ModelMLP.h5")
    
    assert os.path.isfile("tests/testModel/ModelMLP.h5")

def test_loadModel():
    global modelData

    model1 = MultiLayerPerceptron(modelData)
    model1.buildModel(
        nHiddenLayers=2,
        neuronsPerLayerList=[10,10],
        activation="sigmoid",
        optimizer="SGD"
    )
    model1.saveModel("tests/testModel/ModelMLP1.h5")
    
    model2 = MultiLayerPerceptron(modelData)
    model2.buildModel()
    model1.saveModel("tests/testModel/ModelMLP2.h5")

    modelLoaded1 = MultiLayerPerceptron(modelData)
    modelLoaded1.loadModel("tests/testModel/ModelMLP1.h5")
    

    assert isinstance(model2.model, TensorflowModel)
    assert model1 == modelLoaded1
    assert model2 != modelLoaded1





















