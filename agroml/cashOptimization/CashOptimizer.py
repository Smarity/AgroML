from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from sklearn.model_selection import KFold

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
            split = SplitRandom(
                data=self.xTrain, 
                testSize=self.validationSize, 
                randomSeed = self.randomSeed)

            self.xTrain, self.xVal = split.splitToTrainTest()
            self.xTrainIndex = self.xTrain.index
            self.xValIndex = self.xVal.index
            self.xTrain = self._reshapeTwoDimArrayToThreeDimNumpyArray(self.xTrain)
            self.xVal   = self._reshapeTwoDimArrayToThreeDimNumpyArray(self.xVal)

            self.yTrain, self.yVal = split.splitToTrainTest()
            self.yTrainIndex = self.yTrain.index
            self.yValIndex = self.yVal.index
            self.yTrain = self._reshapeTwoDimArrayToThreeDimNumpyArray(self.yTrain)
            self.yVal   = self._reshapeTwoDimArrayToThreeDimNumpyArray(self.yVal)

        elif "Seq" in self.splitFunction:
            split = SplitSequentially(
                data = self.modelData.data.pandasDataFrame,
                testSize = self.validationSize)

            self.xTrain, self.xVal = split.splitToTrainTest()
            self.xTrainIndex = self.xTrain.index
            self.xValIndex = self.xVal.index
            self.xTrain = self._reshapeTwoDimArrayToThreeDimNumpyArray(self.xTrain)
            self.xVal   = self._reshapeTwoDimArrayToThreeDimNumpyArray(self.xVal)

            self.yTrain, self.yVal = split.splitToTrainTest()
            self.yTrainIndex = self.yTrain.index
            self.yValIndex = self.yVal.index
            self.yTrain = self._reshapeTwoDimArrayToThreeDimNumpyArray(self.yTrain)
            self.yVal   = self._reshapeTwoDimArrayToThreeDimNumpyArray(self.yVal)


        elif  "Cross" in self.splitFunction:
            kf = KFold(n_splits=self.nFolds, random_state=self.randomSeed)
            
            # the list will be later transformed into numpy array in order to
            # check te shape (fold, nSamples, nFeatures)
            xTrain, yTrain = list(), list()
            xVal, yVal = list(), list()
            for train_index, val_index in kf.split(self.xTrain):
                xTrain.append(self.xTrain[train_index])
                xVal.append(self.xTrain[val_index])
                yTrain.append(self.yTrain[train_index])
                yVal.append(self.yTrain[val_index])

            self.xTrain = np.array(xTrain)
            self.yTrain = np.array(yTrain)
            self.xVal   = np.array(xVal)
            self.yVal   = np.array(yVal)

        else:
            self.xTrain = np.array(self.xTrain)
            self.yTrain = np.array(self.yTrain)
            self.xVal   = np.array(self.xVal)
            self.yVal   = np.array(self.yVal)

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

