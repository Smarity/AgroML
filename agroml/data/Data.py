import os

import pandas as pd
from icecream import ic

class Data:

    def __init__(self, fileLocation:str):
        self.fileLocation = fileLocation
        assert self._doesFileExists()

        self.data = self._readData()
        # get attributes from data
        self.columnNamesList = list(self.data.columns)
        self.nColumns = self.data.shape[1]

    def _doesFileExists(self) -> bool:
        return os.path.exists(self.fileLocation)

    def _getFileExtension(self) -> str:
        return os.path.splitext(self.fileLocation)[-1]

    def _readData(self) -> pd.DataFrame:
        # read the data into a pandas dataframe
        if self._getFileExtension() in set([".xls", ".xlsx"]):
            data = self._readDataExcel()
        else: 
            data = self._readDataCsv()
        return data

    def _readDataExcel(self) -> pd.DataFrame:
        return pd.read_excel(self.fileLocation)
        
    def _readDataCsv(self) -> pd.DataFrame:
        for sep in [",", ";", r"\s+"]: # r"\s+" must be written this way not to give a warning
            data = pd.read_csv(self.fileLocation, sep=sep)
            nColumns = data.shape[1]
            if nColumns > 1:
                break
        else:
            # the r"" (raw string) is used in the message because \s+ gives a warning
            assert(True, r"""Please, use any of the following separators [",", ";", "\s+"]""")
        return data

    # Note: Not implemented yet
    def __repr__(self) -> str:
        Output= """
            "fileLocation": {},
            "nColumns": {},
            "nRows": self.data.shape[0],
            "columnNamesList": {},
            "data.head()": {},
            """.format(self.fileLocation, self.nColumns, self.columnNamesList, self.data.head())

        return Output

    # Note: Not implemented yet
    def __str__(self) -> str:
        Output = {
            "fileLocation": self.fileLocation,
            "nColumns": self.nColumns,
            "nRows": self.data.shape[0],
            "columnNamesList": self.columnNamesList,
            "data.head()": self.data.head(),
        }
        return str(Output)

    def filterFeatures(self, variableList:list) -> pd.DataFrame:
        return self.data.filter(items=variableList)

    @property
    def shape(self) -> tuple:
        return self.data.shape

    @property
    def pandasDataFrame(self) -> pd.DataFrame:
        return self.data