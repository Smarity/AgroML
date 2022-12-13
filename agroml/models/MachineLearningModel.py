from abc import ABC, abstractmethod

from agroml.data import ModelData

class MachineLearningModel(ABC):
    def __init__(self, modelData:ModelData):
        self.modelData = modelData

        self.featuresList = self.modelData.inputList
        self.nFeatures = len(self.featuresList)
        self.outputsList = self.modelData.outputList
        self.nOutputs = len(self.outputsList)

        self.xTrain = self.modelData.xTrain
        self.xTest = self.modelData.xTest
        self.yTrain = self.modelData.yTrain
        self.yTest = self.modelData.yTest

    def __str__():
        pass

    @abstractmethod
    def buildModel(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def optimizationAlgorithm(self):
        pass

    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def saveModel(self):
        pass

    @abstractmethod
    def loadModel(self):
        pass

    @abstractmethod
    def savePredictions(self):
        pass

    
