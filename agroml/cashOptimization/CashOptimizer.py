from abc import ABC, abstractmethod
from typing import Optional

import pdb
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from icecream import ic

from agroml.data import ModelData
from agroml.preprocessing import SplitRandom, SplitSequentially

class CashOptimizer(ABC):

    @abstractmethod
    def __init__(
        self,
        modelData:ModelData,
        splitFunction:Optional[str] = "None",
        validationSize:Optional[float] = 0.2,
        randomSeed:Optional[int] = 42,
        nFolds:Optional[int] = 5,
    ):
        """Base class for cash optimization algorithms.

        Parameters
        ----------
        modelData : ModelData
            ModelData object containing data, input and output variables. The same as the model
            data used in the model.
        splitFunction : Optional[str], optional
            The function to use for splitting the data into train and validation sets, by default "None".
            Any from the following: "None", "SplitRandom", "SplitSequentially", "CrossValidation"
        validationSize : Optional[float], optional
            The proportion of the data to include in the validation set, by default 0.2
        randomSeed : Optional[int], optional
            The seed to use for the random number generator, by default 42
        nFolds : Optional[int], optional
            The number of folds to use in the cross validation, by default 5. Only used if splitFunction
            is "CrossValidation"
        """
        self.modelData = modelData
        self.splitFunction = splitFunction
        self.validationSize = validationSize
        self.randomSeed = randomSeed
        self.nFolds = nFolds

        self.xTrain = modelData.xTrain.reset_index(drop=True)
        self.yTrain = modelData.yTrain.reset_index(drop=True)
        self.xVal   = None
        self.yVal   = None



    def _splitToValidation(self):

        if "Random" in self.splitFunction:
            split_xTrain = SplitRandom(
                data=self.xTrain, 
                testSize=self.validationSize, 
                randomSeed = self.randomSeed)
            self.xTrain, self.xVal = split_xTrain.splitToTrainTest()
            self.xTrainIndex = self.xTrain.index
            self.xValIndex = self.xVal.index
            self.xTrain = self._reshapeTwoDimArrayToThreeDimNumpyArray(self.xTrain)
            self.xVal   = self._reshapeTwoDimArrayToThreeDimNumpyArray(self.xVal)
                
            split_yTrain = SplitRandom(
                data=self.yTrain, 
                testSize=self.validationSize, 
                randomSeed = self.randomSeed)
            self.yTrain, self.yVal = split_yTrain.splitToTrainTest()
            self.yTrainIndex = self.yTrain.index
            self.yValIndex = self.yVal.index
            self.yTrain = self._reshapeTwoDimArrayToThreeDimNumpyArray(self.yTrain)
            self.yVal   = self._reshapeTwoDimArrayToThreeDimNumpyArray(self.yVal)

        elif "Seq" in self.splitFunction:
            split_xTrain = SplitSequentially(
                data =  self.xTrain,
                testSize = self.validationSize)
            self.xTrain, self.xVal = split_xTrain.splitToTrainTest()
            self.xTrainIndex = self.xTrain.index
            self.xValIndex = self.xVal.index
            self.xTrain = self._reshapeTwoDimArrayToThreeDimNumpyArray(self.xTrain)
            self.xVal   = self._reshapeTwoDimArrayToThreeDimNumpyArray(self.xVal)

            split_yTrain = SplitSequentially(
                data=self.yTrain, 
                testSize = self.validationSize)
            self.yTrain, self.yVal = split_yTrain.splitToTrainTest()
            self.yTrainIndex = self.yTrain.index
            self.yValIndex = self.yVal.index
            self.yTrain = self._reshapeTwoDimArrayToThreeDimNumpyArray(self.yTrain)
            self.yVal   = self._reshapeTwoDimArrayToThreeDimNumpyArray(self.yVal)

        elif  "Cross" in self.splitFunction:
            kf = KFold(n_splits=self.nFolds)
            
            # the list will be later transformed into numpy array in order to
            # check te shape (fold, nSamples, nFeatures)
            xTrain = list()
            yTrain = list()
            xVal = list()
            yVal = list()

            # Kfold does not give always the same number of samples in the validation set
            # and training set. This is why we calculate the minimum len
            nSamplesTrain = list()
            nSamplesVal = list()
            for train_index, val_index in kf.split(self.xTrain):
                nSamplesTrain.append(len(train_index))
                nSamplesVal.append(len(val_index))
            minTrain = min(nSamplesTrain)
            minVal = min(nSamplesVal)
             
            # Calculate the train and validation sets
            for train_index, val_index in kf.split(self.xTrain):
                xTrain.append(self.xTrain.iloc[train_index[:minTrain]])
                yTrain.append(self.yTrain.iloc[train_index[:minTrain]])
                xVal.append(self.xTrain.iloc[val_index[:minVal]])
                yVal.append(self.yTrain.iloc[val_index[:minVal]])
          
            
            self.xTrain = np.array(xTrain)
            self.yTrain = np.array(yTrain)
            self.xVal = np.array(xVal)
            self.yVal = np.array(yVal)

        else:
            self.xTrain = np.array(self.xTrain)
            self.xTrain = self._reshapeTwoDimArrayToThreeDimNumpyArray(self.xTrain)

            self.yTrain = np.array(self.yTrain)
            self.yTrain = self._reshapeTwoDimArrayToThreeDimNumpyArray(self.yTrain)
            
            self.xVal   = None
            self.yVal   = None

    def _reshapeTwoDimArrayToThreeDimNumpyArray(self, array)->np.array:
        npArray = np.array(array)
        npArray = np.reshape(npArray, (1, npArray.shape[0], npArray.shape[1]))
        return npArray

    @abstractmethod
    def __str__():
        pass

    @abstractmethod
    def __repr__():
        pass

    @abstractmethod
    def __eq__():
        pass

    @abstractmethod
    def optimize():
        pass

