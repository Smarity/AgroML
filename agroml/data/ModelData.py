import warnings
from typing import Optional

import pytest
import pandas as pd
from icecream import ic

from agroml.data import Data
from agroml.preprocessing import SplitRandom, SplitSequentially, SplitByYear, SplitByStation
from agroml.preprocessing import StandardScaler, MinMaxScaler
from agroml.utils import doNotRunItTwice

class ModelData():
    """ 
    A class to be used inside the Machine learning models.
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

        self.inputData, self.outputData = self.defineInputAndOutputData(self.data)

        # if not all features exist, raise a warning and update the features lists
        if not(self._checkAllFeaturesExist()):
            warnings.warn("Some features do not exist in the data",UserWarning)
            self._updateFeaturesLists()

        self._errorNoInputOrOutputFeatures()

    def defineInputAndOutputData(self, data):
        """ It defines the input and output data based on the input and output features

        Parameters
        ----------
        data : Data or pd.DataFrame
        
        Returns
        -------
        (pd.DataFrame, pd.DataFrame)
            The input and output data
        """
        if type(data) is Data:
            inputData = data.filterFeatures(self.inputList)
            outputData = data.filterFeatures(self.outputList)
        elif type(data) is pd.DataFrame:
            inputData = data[self.inputList]
            outputData = data[self.outputList]

        return inputData, outputData

    def _checkAllFeaturesExist(self):
        for feature in self.allFeaturesList:
            if feature not in self.allDataFeaturesList:
                return False
        return True

    def _updateFeaturesLists(self):
        ic(self.inputData.columns)
        self.inputList = list(self.inputData.columns)
        self.outputList = list(self.outputData.columns)
        self.allFeaturesList = self.inputList + self.outputList
                
    def _errorNoInputOrOutputFeatures(self):
        assert(len(self.inputList) == 0, "No input features")
        assert(len(self.outputList) == 0, "No output features")
    
    def splitToTrainTest(
        self, 
        splitFunction:str="SplitRandom",
        testSize:Optional[float]=0.2,
        randomState:Optional[int]=42,
        year:Optional[int]=2016,
    ):
        """ It splits the data into train and test data. Besides, it obtains the xTrain, yTrain, xTest, yTest data
        
        Parameters
        ----------
        splitFunction : str, optional
            The function to split the data ["SplitRandom", "SplitSequentially", "SplitByYear", "SplitByStation"], by default "SplitRandom"
        testSize : Optional[float], optional
            The size of the test data. Just for SplitRandom and SplitSequentially, by default 0.2 
        
        """
        if splitFunction == "SplitRandom":
            splitFunction = SplitRandom(
                data=self.data.pandasDataFrame, 
                testSize=testSize, 
                randomState=randomState,
            )
        elif splitFunction == "SplitSequentially":
            splitFunction = SplitSequentially(
                data = self.data.pandasDataFrame,
                testSize = 0.3,
            )
        elif splitFunction == "SplitByYear":
            splitFunction = SplitByYear(
                data = self.data.pandasDataFrame,
                year = year,
            )
        else:
            warnings.warn(UserWarning("Your split function is not defined"))
            splitFunction = SplitRandom(data=self.data.pandasDataFrame)

        self._dataTrain, self._dataTest = splitFunction.splitToTrainTest()

        # split the train and test into xTrain, yTrain, xTest, yTest
        self._xTrain, self._yTrain = self.defineInputAndOutputData(self._dataTrain)
        self._xTest, self._yTest = self.defineInputAndOutputData(self._dataTest)


    # To be implemented:
    #   - Avoid two normalization for data
    @doNotRunItTwice
    def normalizeData(self, method:str="StandardScaler"):
        """ It normalizes the data in the dataTrain and dataTest

        Parameters
        ----------
        method : str, optional
            The method to normalize the data ["StandardScaler", "MinMaxScaler"], by default "StandardScaler"
        """
        # If the data has not been split, we split it in the default way
        if not(hasattr(self, "_dataTrain")):
            warnings.warn(UserWarning("You have to split the data before normalizing it. The default split has been used"))
            self.splitToTrainTest()

        if "Standard" or "standard" in method:
            self._scaler = StandardScaler(xTrain = self._xTrain)
        else:
            self._scaler = MinMaxScaler(xTrain = self._xTrain)

        self._scaler.fit() # Not needed, anyway
        self._xTrain = self._scaler.transform(self._xTrain)
        self._xTrain = pd.DataFrame(self._xTrain, columns=self.inputList)

        self._xTest = self._scaler.transform(self._xTest)
        self._xTest = pd.DataFrame(self._xTest, columns=self.inputList)        

    def saveScaler(self, path:str):
        """ It saves the scaler

        Parameters
        ----------
        path : str
            The path to save the scaler
        """
        if not(hasattr(self, "_scaler")):
            warnings.warn("You have to normalize the data before saving the scaler. The default normalization has been used", UserWarning)
            self.normalizeData()

        self._scalerPath = path
        self._scaler.save(path)
    

    @property
    def xTrain(self):
        return self._denormalizeData(self._xTrain)
    @property
    def yTrain(self):
        return self._yTrain # no transformation for y data
    @property
    def xTest(self):
        return self._denormalizeData(self._xTest)
    @property
    def yTest(self):
        return self._yTest # no transformation for y data
    @property
    def dataTrain(self):
        return self._dataTrain
    @property
    def dataTest(self):
        return self._dataTest
   
    def _denormalizeData(self, dataFrame: pd.DataFrame):
        """ It denormalizes the data

        Parameters
        ----------
        data : pd.DataFrame
            The data to denormalize

        Returns
        -------
        pd.DataFrame
            The denormalized data if the data has been normalized before
        """
        if hasattr(self, "_scaler"):
            output = pd.DataFrame(self._scaler.inverseTranform(dataFrame), columns=dataFrame.columns)
            return output
        return dataFrame

        
        