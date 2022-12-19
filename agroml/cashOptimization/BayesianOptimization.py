import pandas as pd
from typing import Optional

from agroml.data import ModelData
from agroml.cashOptimization import CashOptimizer

class BayesianOptimization(CashOptimizer):
    def __init__(
        self, 
        modelData:ModelData,
        splitFunction:Optional[str] = "None",
        validationSize:Optional[float] = 0.2,
        randomSeed:Optional[int] = 42,
        nFolds:Optional[int] = 5,
    ):
        super().__init__(
            modelData,
            splitFunction,
            validationSize,
            randomSeed,
            nFolds,
        )  
        
        super()._splitToValidation()

    def __str__(self):
        pass

    def __repr__(self):
        pass

    def __eq__(self):
        pass

    def optimize(self):
        pass