import os

import pandas as pd

class Data:

    def __init__(self, fileLocation:str):
        self.fileLocation = fileLocation

        assert self._doesFileExists()

        # read the data into a pandas dataframe
        if self._getFileExtension() in set([".xls", ".xlsx"]):
            self.readDataExcel()
        else: 
            self.readDataCsv()

        # get attributes from data
        self.columnNamesList = list(self.data.columns)
        self.nColumns = self.data.shape[1]


    def _doesFileExists(self) -> bool:
        return os.path.exists(self.fileLocation)

    def _getFileExtension(self) -> str:
        return os.path.splitext(self.fileLocation)[-1]

    def readDataExcel(self) -> pd.DataFrame:
        self.data = pd.read_excel(self.fileLocation)
        
        
    def readDataCsv(self) -> pd.DataFrame:
        for sep in [",", ";", r"\s+"]: # r"\s+" must be written this way not to give a warning
            self.data = pd.read_csv(self.fileLocation, sep=sep)
            self.nColumns = self.data.shape[1]
            if self.nColumns > 1:
                break
        else:
            # the r"" (raw string) is used in the message because \s+ gives a warning
            assert(True, r"""Please, use any of the following separators [",", ";", "\s+"]""")


