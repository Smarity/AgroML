from agroml.data import Data

class ModelData():

    def __init__(self, Data:Data):
        self.Data = Data
        self.allInputFeaturesList = self.Data.columnNamesList

    def defineInputsAndOutputsFromInputData(self, inputList:list, outputList:list):
        self.input