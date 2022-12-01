import os

import pandas as pd

class InputData:

    def __init__(self, fileLocation):
        assert self._doesFileExists(fileLocation), "File does not exist"

    def _doesFileExists(fileLocation) -> bool:
        return os.path.exists(fileLocation)

    def _getFileExtension(fileLocation) -> str:
        return os.path.splitext(fileLocation)[-1]

