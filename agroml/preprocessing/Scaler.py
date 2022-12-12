import os
import warnings
from abc import ABC, abstractmethod

import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from pickle import dump, load

class Scaler(ABC):
    def __init__(
        self, 
        xTrain:pd.DataFrame, 
        scaler = StandardScaler(),
        path:str = None):
        """
        Parameters
        ----------
        xTrain : pd.DataFrame
            The data to be scaled
        typeScaler : sklearn.preprocessing
        """
        self.xTrain = xTrain
        self.scaler = scaler

        if path is not None and self._doesScalerExists(path):
            self.load(path)
        elif path is not None and not(self._doesScalerExists(path)):
            warnings.warn(UserWarning("The scaler does not exist"))
            self.fit()
        else:
            self.fit()

    def _doesScalerExists(self, path) -> bool:
        return os.path.exists(path)
   
    def fit(self):
        self.scaler = self.scaler.fit(self.xTrain)

    def transform(self, data) -> pd.DataFrame:
        dataScaled = self.scaler.transform(data)
        dataScaled = pd.DataFrame(dataScaled, columns=self.xTrain.columns)
        return dataScaled

    def inverseTranform(self, dataScaled) -> pd.DataFrame:
        data = self.scaler.inverseTranform(dataScaled)
        data = pd.DataFrame(data, columns=self.xTrain.columns)
        return data

    def load(self, path:str):
        """
        Parameters
        ----------
        path : str
            The path to the scaler. It must have .pkl extension
        """
        if not path.endswith(".pkl"):
            raise ValueError("The path must have .pkl extension")
        
        self.scaler = load(open(path, "rb"))

    def save(self, path:str):
        """
        Parameters
        ----------
        path : str
            The path to the scaler. It will be saved with the .pkl extension
        """
        if not path.endswith(".pkl"):
            path = path + ".pkl"
        dump(self.scaler, open(path, "wb"))


    

