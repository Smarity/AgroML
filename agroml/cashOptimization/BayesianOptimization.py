import pandas as pd
import numpy as np
from icecream import ic
from typing import Optional

from skopt import gp_minimize
from skopt.utils import use_named_args
from skopt.space import Real, Categorical, Integer

from agroml.data import ModelData
from agroml.models import MachineLearningModel
from agroml.cashOptimization import CashOptimizer
from agroml.utils import getMeanAbsoluteError

class BayesianOptimization(CashOptimizer):
    def __init__(
        self, 
        modelData:ModelData,
        splitFunction:Optional[str] = "None",
        validationSize:Optional[float] = 0.3,
        randomSeed:Optional[int] = 43,
        nFolds:Optional[int] = 6,
        totalEpochs:Optional[int] = 51,
        randomEpochs:Optional[int] = 41
        ):

        super().__init__(
            modelData,
            splitFunction,
            validationSize,
            randomSeed,
            nFolds,
            totalEpochs,
            randomEpochs,
        )  
        
        super()._splitToValidation()

    def __str__(self):
        pass

    def __repr__(self):
        pass

    def __eq__(self):
        pass

    def optimize(
        self,
        mlModel: MachineLearningModel,
        hyperparameterSpace: dict(),
        verbose: int = 0,
        ):

        super().optimize(
            mlModel,
            hyperparameterSpace,
        )           

        # set the hyperparameter dimension
        self.hyperparameterDimensionList =  [
            hyperparameterSpace["activation"],
            hyperparameterSpace["optimizer"],
            hyperparameterSpace["epochs"],
        ]

        for i in range(hyperparameterSpace["nHiddenLayers"].high):
            name_neuron = 'neurons_' + str(i+1)
            self.hyperparameterDimensionList.append(
                Integer(
                    low=1, 
                    high=hyperparameterSpace["nHiddenLayers"].high,
                    name=name_neuron
                )
            )

        # set the hyperparameter default values
        self.hyperparameterDefaultValuesList = [
            dimension.low 
            if (type(dimension) is Integer or type(dimension) is Real)
            else dimension.categories[0]
            for dimension in self.hyperparameterDimensionList
        ]
        

        # fitness function
        @use_named_args(dimensions= self.hyperparameterDimensionList)
        def _fitnessFunction(**params):
            ic(params)
            # build model
            self.model = self.mlModel.buildModel(
                nHiddenLayers = params['nHiddenLayers'], 
                neuronsPerLayerList=params['neuronsPerLayerList'],
                activation=params['activation'],
                optimizer=params['optimizer'],
                epochs=params['epochs']
            )

            # fit model
            maeList = list()
            for i in range(self.nFolds):
                # set datasets in the model
                self.mlModel.xTrain = self.xTrain[i]
                self.mlModel.yTrain = self.yTrain[i]

                if self.xVal is not None:
                    self.mlModel.xTest = self.xVal[i]
                    self.mlModel.yTest = self.yVal[i]
                else:
                    self.mlModel.xTest = self.xTrain[i]
                    self.mlModel.yTest = self.yTrain[i]

                self.mlModel.train(
                    verbose=self.verbose,
                )
            
                yPred = self.mlModel.predict()
                maeList.append(getMeanAbsoluteError(self.yTrain[i], self.yPred))
                
            return np.mean(maeList)


        # Bayesian optimization
        self.bestParams = gp_minimize(
            func = _fitnessFunction,       
            dimensions = self.hyperparameterDimensionList, 
            n_calls = self.totalEpochs,     
            n_jobs = -1,
            x0 = self.hyperparameterDefaultValuesList,
            n_random_starts = self.randomEpochs)
        self.bayesianIterations = self.bestParams.x_iters
        
        # best model
        self.bestModel = self.buildModel(
            hiddenLayers = self.bestParams.x[0],
            neurons = self.bestParams.x[1],
            activation= self.bestParams.x[2],
            optimizer = self.bestParams.x[3])
        
        # best parameters
        self.bestParams = {
            'nHiddenLayers': self.bestParams.x[0], 
            'neuronsPerLayerList': self.bestParams.x[1],
            'activation': self.bestParams.x[2],
            'optimizer': self.bestParams.x[3],
            'epochs': self.bestParams.x[4]
        }

        return self.bestModel, self.bestParams