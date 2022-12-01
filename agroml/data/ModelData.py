import warnings
from typing import Optional

import pytest
from icecream import ic

from agroml.data import Data
from agroml.splitFunctions import SplitRandom, SplitNotRandom

class ModelData():
    """ A class to be used inside the Machine learning models.
    It contains the data and the input and output features.


    
    """

    def __init__(self, data:Data, inputList:list, outputList:list):
        # data attributes
        self.data = data
        self.allDataFeaturesList = self.data.columnNamesList
        # features attributes
        self.inputList = inputList
        self.outputList = outputList
        self.allFeaturesList = self.inputList + self.outputList

        self.defineInputAndOutputData()

        # if not all features exist, raise a warning and update the features lists
        if not(self._checkAllFeaturesExist()):
            warnings.warn(UserWarning("Some features do not exist in the data"))
            self._updateFeaturesLists()

        self._errorNoInputOrOutputFeatures()

    def defineInputAndOutputData(self):
        self.inputData = self.data.filterFeatures(self.inputList)
        self.outputData = self.data.filterFeatures(self.outputList)

    def _checkAllFeaturesExist(self):
        for feature in self.allFeaturesList:
            if feature not in self.allDataFeaturesList:
                return False
        return True

    def _updateFeaturesLists(self):
        self.inputList = list(self.inputData.columns)
        self.outputList = list(self.outputData.columns)
        self.allFeaturesList = self.inputList + self.outputList
                
    def _errorNoInputOrOutputFeatures(self):
        assert(len(self.inputList) == 0, "No input features")
        assert(len(self.outputList) == 0, "No output features")

    
    def splitToTrainTest(
        self, 
        splitFunction:Optional[str]="SplitRandom",
        testSize:Optional[float]=0.2,
        randomState:Optional[int]=42,
        year:Optional[int]=None
    ):
        if splitFunction == "SplitRandom":
            split = SplitRandom(data=self.data, testSize=testSize, randomState=randomState)
        elif splitFunction == "SplitNotRandom":
            split = SplitNotRandom(testSize)
        else:
            warnings.warn(UserWarning("Your split function is not defined"))
            pass
        
        pass
