from abc import ABC, abstractmethod

import pandas as pd
from sklearn.model_selection import train_test_split

class Split(ABC):
    @abstractmethod
    def splitToTrainTest(self):
        pass
    
class SplitRandom(Split):
    def splitToTrainTest(data:pd.DataFrame, testSize:float=0.3, randomState:int=42) -> tuple:
        """Split the data into train and test sets randomly.

        Parameters
        ----------
        data : pd.DataFrame
            The data to split
        testSize : float, optional
            The proportion of the data to include in the test set, by default 0.3
        randomState : int, optional
            The seed to use for the random number generator, by default 42

        Returns
        -------
        tuple
            The train and test sets
        """
        train, test = train_test_split(data, test_size=testSize, random_state=randomState)
        return train, test

class SplitNotRandom(Split):
    def splitToTrainTest(data:pd.DataFrame, trainSize:float=0.7) -> tuple:
        """Split the data into train and test sets not randomly.

        Parameters
        ----------
        data : pd.DataFrame
            The data to split
        trainSize : float, optional
            The proportion of the data to include in the train set, by default 0.7

        Returns
        -------
        tuple
            The train and test sets
        """
        nRows = data.shape[0]
        nTrainRows = int(nRows * trainSize)
        train = data.iloc[:nTrainRows, :]
        test = data.iloc[nTrainRows:, :]
        return train, test


