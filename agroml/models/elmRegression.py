
import skopt
from skopt import gp_minimize
from skopt.utils import use_named_args
from skopt.space import Real, Categorical, Integer 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_squared_error as MSE
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import hpelm 
import random
from functools import lru_cache 
from sklearn.model_selection import KFold

from agroml.utils.statistics import *

class ExtremeLearningMachine:
    def __init__(
        self, 
        xTrain, 
        xTest, 
        yTrain, 
        yTest):
        """
        Arguments:
            xTrain {array or np.array} - shape(bath, lagDays, nFeaturesInput)

            xTest {array} - shape(bath, lagDays, nFeaturesInput)

            yTrain {array} - shape(bath, nFeaturesOutput)

            yTest {array} - shape(bath, nFeaturesOutput)
        """
        # input data as float
        try:
            self.xTrain = np.array(xTrain).astype(float)
            self.xTest = np.array(xTest).astype(float)
            self.yTrain = np.array(yTrain).astype(float)
            self.yTest = np.array(yTest).astype(float)
        except:
            print("There exist non-numeric values")
            raise
        
        # shape is not correct
        assert len(self.xTrain.shape)==3, 'xTrain shape is wrong'
        assert len(xTest.shape)==3, 'xTrain shape is wrong'
        assert len(yTrain.shape)==2, 'yTrain shape is wrong'
        assert len(yTest.shape)==2, 'yTrain shape is wrong'
        
        # all lagDays must be the same
        assert xTrain.shape[1]==xTest.shape[1]
        
        self.nInputs = self.xTrain.shape[2]
        self.nOutputs = self.yTrain.shape[1]
        self.lagDays = self.xTrain.shape[1]
        self.batchTrain = self.xTrain.shape[0]
        self.batchTest = self.xTest.shape[0]
        
        
        # reshape x data - hpelm does no accept 3D or higher dimensional
        self.xTrain = np.reshape(self.xTrain,(self.batchTrain,self.lagDays*self.nInputs))
        self.xTest = np.reshape(self.xTest,(self.batchTest,self.lagDays*self.nInputs))
        
    def buildModel(
        self, 
        neurons, 
        activation):
        """
        It builds the model using the following parameters
        
        Arguments:
            neurons {int} - Number of neurons in the hidden layer.

            activation {str} - Activation function ['lin', 'sigm', 
            'tanh', 'rbf_l1', 'rbf_l2', 'rbf_linf']
            https://hpelm.readthedocs.io/en/latest/api/elm.html

        Output:
            model {Model} - Compiled model from hpelm
        """        
        model = hpelm.ELM(inputs=self.nInputs*self.lagDays, outputs=self.nOutputs)
        model.add_neurons(number=int(neurons), func=activation)
        self.model = model
        return model

    def saveModel(self, model, fileName):
        """
        It saves the model in an specific location and name

        Arguments:
            model {keras.model} - Model to save 
            fileName {str} - file name to save model (without extension)
        """
        self.model.save(fileName+'.pkl')

    def loadModel(self, fileName):
        """
        It loads a specific model 

        Arguments:
            fileName {str} - file name to save model (with extension)

        Output:
            model {keras model} - It returns the model
        """
        model = hpelm.ELM(inputs=self.nInputs*self.lagDays, outputs=self.nOutputs)
        model = model.load(fileName)
        return model

    def savePredictions(self, yPred, yTest):
        pass
        
    def trainFullTrainingData(
        self, 
        model,
        *args, 
        **kwargs):
        '''
        It trains the model, on the full training dataset

        Arguments:
            model {hpelm.elm.ELM object}
        '''
        model = model.train(self.xTrain, self.yTrain, 'v','OP', 'r')
            
        return model
        
    def predictFullTestingData(
        self, 
        model):
        '''
        It makes predictions on the testing dataset
        '''
        pred = np.array(self.model.predict(self.xTest))
        pred = pred.reshape(self.batchTest, self.nOutputs)
        return pred   
        
    def trainModel(
        self, 
        model, 
        xTrain, 
        yTrain, 
        *args, 
        **kwargs):
        '''
        It trains the model on a specific training dataset

        Arguments:
            model {hpelm.elm.ELM object} - model

            xTrain {np.array} - input training data with shape (bath, lagDays, nFeaturesInput)

            yTrain {np.array} - output training data with shape (bath, nFeaturesOutput)
        '''
        
        model = self.model.train(xTrain, yTrain, 'v','OP', 'r')
            
        return model
    
    def predictModel(
        self, 
        model, 
        xTest):
        '''
        The model makes predictions on a specific dataset
        
        Arguments:
            model
            xTest (np.array) - InputData to predict
        '''
        pred = np.array(self.model.predict(xTest))
        pred = pred.reshape(xTest.shape[0], self.nOutputs)
        return pred   
    
    def _fitAndValidationAssess(
        self, 
        model, 
        validationSplit, 
        shuffle, 
        crossVal, 
        nFolds):
        """
        It splits the training data into validation using different techniches, holdout and crossvalidation.
        In the holdout
        """
        
        if crossVal:
            kf = KFold(n_splits=nFolds)
            maeList = []
            for train_index, val_index in kf.split(self.xTrain):
                bayes_xTrain = self.xTrain[train_index]
                bayes_xVal = self.xTrain[val_index]
                bayes_yTrain = self.yTrain[train_index]
                bayes_yVal = self.yTrain[val_index]
                
                model = self.trainModel(model, bayes_xTrain, bayes_yTrain)
                bayes_yPred = self.predictModel(model, bayes_xVal)
                
                mae = getMeanAbsoluteError(bayes_yVal, bayes_yPred)
                mae = np.min(mae)
                maeList.append(mae)

            mae_np = np.array(maeList)
            mae = np.mean(mae_np)

            return mae

        elif validationSplit == 0.0:
            model = self.trainFullTrainingData(model)
            yPred = self.predictFullTestingData(model)
            mae = getMeanAbsoluteError(self.yTest, yPred)
            mae = np.min(mae)

            return mae

        else:
            if shuffle==True:
                bayes_xTrain, bayes_xVal, bayes_yTrain, bayes_yVal = train_test_split(
                    self.xTrain,
                    self.yTrain,
                    random_state = 42,
                    test_size = validationSplit)
            else:
                # validation index
                validationIndex = int((1-validationSplit)*len(self.xTrain))
                # get training data to validation
                if self.nInputs == 1:
                    bayes_xTrain = self.xTrain[:validationIndex]
                    bayes_xVal = self.xTrain[validationIndex:]
                else:
                    bayes_xTrain = self.xTrain[:validationIndex,:]
                    bayes_xVal = self.xTrain[validationIndex:,:]
                # get validation data
                if self.nOutputs == 1:
                    bayes_yTrain = self.yTrain[:validationIndex]
                    bayes_yVal = self.yTrain[validationIndex:]
                else:
                    bayes_yTrain = self.yTrain[:validationIndex,:]
                    bayes_yVal = self.yTrain[validationIndex:,:]

            model = self.trainModel(model, bayes_xTrain, bayes_yTrain)
            bayes_yPred = self.predictModel(model, bayes_xVal)
            mae = getMeanAbsoluteError(bayes_yVal, bayes_yPred)
            mae = np.min(mae)
            
            return mae
        
    def bayesianOptimization(
        self, 
        neuronsList=[5, 25], 
        activationList=['lin'], 
        bayesianEpochs=50, 
        randomStart=40, 
        validationSplit=0.0, 
        shuffle=False, 
        crossVal= False, 
        nFolds = 4):
        '''
        It tunes the different hyperparameters using bayesian optimization
        
        Arguments:
            neuronsList {list of ints} - Maximum number of neurons in a hidden layer.

            activationList {list of int) - list with the activation functions to test
                ['lin','sigm','tanh','rbf_l1','rbf_l2','rbf_linf']

            bayesianEpochs {int) - number of total epochs in Bayesian Optimization

            randomStart {int) - number of random epochs in Bayesian Optimization

            validationSplit {float} (from 0 to 1) with the percentaje of validation dataset

            shuffle=False {bool} - If True, it shuffles the data of validation split
                (only on holdout, not crossVal)

            crossVal {bool} - If True, it carries out a cross validation

            nFolds {int} - Number of folds in cross validation
        '''
        
        # creation of the hyperparameter space
        dim_neurons = Integer(low=neuronsList[0], high=neuronsList[1], name='neurons')
        dim_activation = Categorical(categories=activationList, name='activation')
        
        self.dimensions = [ dim_neurons, dim_activation]
        
        # default parameters
        self.defaultParameters = [
            neuronsList[0], activationList[0]
        ]
        
        # fitness function
        @use_named_args(dimensions= self.dimensions)
        def fitnessFunction(**params):
        
            # build model
            model = self.buildModel(
                neurons=params['neurons'],
                activation=params['activation'])
        
            mae = self._fitAndValidationAssess(model, validationSplit, shuffle, crossVal, nFolds)
            
            return mae
        
            # Bayesian optimization
        bayesianBestParameters = gp_minimize(
            func=fitnessFunction,       # the function to minimize
            dimensions=self.dimensions,  # the bounds on each dimension of x
            n_calls=bayesianEpochs,     # the number of evaluations of f
            n_jobs=-1,                   # Number of cores to run in parallel while running
            x0=self.defaultParameters,  #Initial input points
            n_random_starts=randomStart)
        
        self.bayesianIterations = bayesianBestParameters.x_iters
        
        # best model
        model = self.buildModel(
            neurons = bayesianBestParameters.x[0],
            activation= bayesianBestParameters.x[1])
        
        # best parameters
        bestParams = {
            'neurons': bayesianBestParameters.x[0],
            'activation': bayesianBestParameters.x[1]
        }

        return model, bestParams