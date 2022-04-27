import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from skopt.utils import use_named_args
from skopt.space import Real, Categorical, Integer
from skopt import gp_minimize
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from icecream import ic
import pickle

from agroml.utils.statistics import *

class RandomForest:
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

        # other assertions
        assert self.yTrain.shape[1] == 1, 'RF does not allow having more than one output'
        assert self.yTest.shape[1] == 1, 'RF does not allow having more than one output'
        
        self.nInputs = self.xTrain.shape[2]
        self.nOutputs = self.yTrain.shape[1]
        self.lagDays = self.xTrain.shape[1]
        self.batchTrain = self.xTrain.shape[0]
        self.batchTest = self.xTest.shape[0]
        
        
        # reshape x data - scikit-learn does no accept 3D or higher dimensional
        self.xTrain = np.reshape(self.xTrain,(self.batchTrain,self.lagDays*self.nInputs))
        self.xTest = np.reshape(self.xTest,(self.batchTest,self.lagDays*self.nInputs))
        
    def buildModel(self, nEstimators, maxFeatures):
        '''
        Arguments:
            nEstimators {int} - The number of trees in the forest.
            max_features -> The number of features to consider when looking for the best split:
                * If int, then consider max_features features at each split.
                * If float, then max_features is a fraction and int(max_features * n_features) 
                features are considered at each split.
                * If “sqrt”, then max_features=sqrt(n_features).
                * If “log2”, then max_features=log2(n_features).
                * If None, then max_features=n_features.
                [1/3, 1/2, 3/4, 'sqrt', 'log2', 'None']
        
        '''
        model = RandomForestRegressor(
            n_estimators=nEstimators,
            criterion='mae',
            max_features=maxFeatures,
            bootstrap=True,
            verbose=0,
            n_jobs=-1)

        return model

    def saveModel(self, model, fileName):
        """
        It saves the model in an specific location and name

        Arguments:
            model {keras.model} - Model to save 
            fileName {str} - file name to save model (without extension)
        """
        pickle.dump(model, open(fileName+'.sav', 'wb'))

    def loadModel(self, fileName):
        """
        It loads a specific model 

        Arguments:
            fileName {str} - file name to save model (with extension)

        Output:
            model {keras model} - It returns the model
        """
        with open(fileName, 'rb') as file:
            model = pickle.load(fileName)
            
        return model
        
    def savePredictions(self, yPred, yTest, fileName):
        """
        It saves the predictions values of a model
        Arguments:
            yPred {array} - Array with the predictions. The shape must be
                (batch, number of output features)
            yTest {array} - Array with the measured values. The shape must be
                (batch, number of output features)
            fileName {str} - file name to save model (with extension)
        Output:
            None
        """
        dfPredictions = pd.DataFrame()

        for i in range(self.nOutputs):
            dfPredictions["pred_{}".format(i)] = yPred[:, i]
            dfPredictions["meas_{}".format(i)] = yTest[:, i]

        dfPredictions.to_csv(str(fileName)+".csv")
        
        
    def trainFullTrainingData(
        self, 
        model, 
        *args, 
        **kwargs):
        '''
        It trains the model on the full training dataset'

        Arguments:
            verbose {int} - 1: Show details / 0: Do not show details
            showGraph {Bool} - True: See training graph / False: Do not see training graph
        '''
        
        model = model.fit(self.xTrain, self.yTrain)

        return model
        
    def predictFullTestingData(
        self, 
        model):
        '''
        It makes predictions based on the full testing dataset
        '''
        pred = np.array(model.predict(self.xTest))
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
        It trains the model on especific training dataset

        Arguments:
            model

            xTrain {np.array}

            yTrain {np.array}
        '''
        model = model.fit(xTrain, yTrain)
  
        return model
    
    def predictModel(
        self, 
        model, 
        xTest):
        '''
        It makes predictions on especific testing dataset
        
        Arguments:
            model
            xTest (np.array) - InputData to predict
        '''
        pred = np.array(model.predict(xTest))
        pred = pred.reshape(xTest.shape[0], self.nOutputs)
        return pred   
    
    def _fitAndValidationAssess(
        self, 
        model, 
        validationSplit, 
        shuffle, 
        crossVal, 
        nFolds, 
        epochs):
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
                
                model = self.trainModel(model, bayes_xTrain, bayes_yTrain, epochs)
                bayes_yPred = self.predictModel(model, bayes_xVal)
                
                mae = getMeanAbsoluteError(bayes_yVal, bayes_yPred)
                mae = np.min(mae)
                maeList.append(mae)

            mae_np = np.array(maeList)
            mae = np.mean(mae_np)
            return mae

        elif validationSplit == 0.0:
            model = self.trainFullTrainingData(model, epochs)
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

            model = self.trainModel(model, bayes_xTrain, bayes_yTrain, epochs)
            bayes_yPred = self.predictModel(model, bayes_xVal)
            mae = getMeanAbsoluteError(bayes_yVal, bayes_yPred)
            mae = np.min(mae)
            
            return mae
        
    def bayesianOptimization(
        self, 
        nEstimatorsList = [10, 150], 
        maxFeaturesList = ['sqrt', 'log2', 'None'],
        bayesianEpochs=50, 
        randomStart=40, 
        validationSplit=0.0, 
        shuffle=False, 
        crossVal= False, 
        nFolds = 4):
        '''
        It tunes the different hyperparameters using bayesian optimization
        
        Arguments:
            nEstimatorsList {list of ints} - [min, max]. 
                Number of trees in one forest 
            maxFeatures {list of floats and/or str} - 
                Number of features to consider when looking for the best split
                [1/3, 1/2, 3/4, 'sqrt', 'log2', 'None']
            bayesianEpochs (int) - number of total epochs in Bayesian Optimization
            randomStart (int) - number of random epochs in Bayesian Optimization
            validationSplit - float (from 0 to 1) with the percentaje of validation dataset
            shuffle=False (bool) - If True, it shuffles the data of validation split
                (only on holdout, not crossVal)
            crossVal (bool) - If True, it carries out a cross validation
            nFolds (int) - Number of folds in cross validation
        '''

        # creation of the hyperparameter space
        # creation of the dimensions of the variables to study
        dim_nEstimators = Integer(low=nEstimatorsList[0], high=nEstimatorsList[1], name='nEstimators')
        dim_maxFeatures = Categorical(categories=maxFeaturesList, name='maxFeatures')
        
        self.dimensions = [dim_nEstimators, dim_maxFeatures]
        
        # default parameters
        self.defaultParameters = [nEstimatorsList[0], maxFeaturesList[0]]
        
        # fitness function
        @use_named_args(dimensions= self.dimensions)
        def fitnessFunction(**params):
            
            try:
                # build model
                model = self.buildModel(
                    nEstimators = params['nEstimators'], 
                    maxFeatures = params['maxFeatures'])
            
                mae = self._fitAndValidationAssess(
                    model, validationSplit, shuffle, crossVal, nFolds, params['epochs'])
                
            except:
                mae=1000
            
            return mae
        
            # Bayesian optimization
        bayesianBestParameters = gp_minimize(
            func=fitnessFunction,        # the function to minimize
            dimensions=self.dimensions,  # the bounds on each dimension of x
            n_calls=bayesianEpochs,      # the number of evaluations of f
            n_jobs=-1,                   # Number of cores to run in parallel while running
            x0=self.defaultParameters,   # Initial input points
            n_random_starts=randomStart)
        
        self.bayesianIterations = bayesianBestParameters.x_iters
        
        # best model
        model = self.buildModel(
            nEstimators = bayesianBestParameters.x[0], 
            maxFeatures = bayesianBestParameters.x[1])
        
        # best parameters
        bestParams = {
            'nEstimators': bayesianBestParameters.x[0], 
            'maxFeatures': bayesianBestParameters.x[1]
        }

        return model, bestParams
        
        
        
        
        
