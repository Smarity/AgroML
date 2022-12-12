from abc import ABC, abstractmethod
from typing import Optional

from agroml.data import ModelData

class Model(ABC):
    def __init__(self, modelData:ModelData):
        self.modelData = modelData
    
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

    
