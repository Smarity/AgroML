from abc import ABC, abstractmethod

import pandas as pd
from sklearn.model_selection import train_test_split

class Split(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def splitToTrainTest(self):
        pass
    
class SplitRandom(Split):
    def __init__(self, data:pd.DataFrame, testSize:float=0.3, randomSeed:int=42) -> None:
        self.data = data
        self.testSize = testSize
        self.randomSeed = randomSeed

    def splitToTrainTest(self) -> tuple:
        """Split the data into train and test sets randomly.

        Parameters
        ----------
        data : pd.DataFrame
            The data to split
        testSize : float, optional
            The proportion of the data to include in the test set, by default 0.3
        randomSeed : int, optional
            The seed to use for the random number generator, by default 42

        Returns
        -------
        (pd.DataFrame, pd.DataFrame)
            A tuple with the train and test pandas dataframe
        """
        
        train, test = train_test_split(self.data, test_size=self.testSize, random_state=self.randomSeed)
        train = pd.DataFrame(train, columns=self.data.columns)
        test = pd.DataFrame(test, columns=self.data.columns)

        return train, test

class SplitSequentially(Split):
    def __init__(self, data:pd.DataFrame, testSize:float=0.3) -> None:
        self.data = data
        self.testSize = testSize

    def splitToTrainTest(self) -> tuple:
        """Split the data into train and test sets not randomly.

        Parameters
        ----------
        data : pd.DataFrame
            The data to split
        testSize : float, optional
            The proportion of the data to include in the test set, by default 0.3

        Returns
        -------
        (pd.DataFrame, pd.DataFrame)
            A tuple with the train and test pandas dataframe
        """
        nRows = self.data.shape[0]
        nTrainRows = int(nRows * (1-self.testSize))
        train = self.data.iloc[:nTrainRows, :]
        test = self.data.iloc[nTrainRows:, :]

        return train, test

class SplitByYear(Split):
    def __init__(self, data:pd.DataFrame, yearTest:int) -> None:
        self.data = data
        self.yearTest = yearTest

    def splitToTrainTest(self) -> tuple:
        """Split the data into train and test sets based on the year from the dataset

        Parameters
        ----------
        data : pd.DataFrame
            The data to split
        yearTest :int
            The starting year of the test set

        Returns
        -------
        (pd.DataFrame, pd.DataFrame)
            A tuple with the train and test pandas dataframe
        """

        pass

class SplitByStation(Split):
    def __init__(self, data:pd.DataFrame, trainStationList:list, testStationList:list) -> None:
        self.data = data
        self.trainStationList = trainStationList
        self.testStationList = testStationList

    def splitToTrainTest(self) -> tuple:
       pass
