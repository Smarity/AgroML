import os

from agroml.utils.splitDataByPercentageWithLagDays import splitDataByPercentageWithLagDays
import pandas as pd
import numpy as np
from icecream import ic

from agroml.utils.splitDataByStation import splitDataByStation
from agroml.utils.splitDataByYear import splitDataByYear
from agroml.models.mlpRegression import MultiLayerPerceptron
from agroml.utils.statistics import *


def test_dataInputTypes():
    # import dataset from example
    df = pd.read_csv('tests/test-data/data-example.csv', sep=';')
    
    # get important variables
    uniqueStations = np.unique(df['station'])
    uniqueYears = np.unique(df['year'])
    varListInputs = ['tx', 'tn', 'rs']
    varListOutputs = ['et0']
    
    # split data to train and test
    xTrain, xTest, yTrain, yTest, scaler = splitDataByYear(
        df=df,
        station=uniqueStations[-1], 
        yearTestStart=uniqueYears[-1], 
        varListInputs=varListInputs, 
        varListOutputs=varListOutputs)
    
    mlpModel = MultiLayerPerceptron(xTrain, xTest, yTrain, yTest)
    
    assert type(mlpModel.xTrain) is np.ndarray
    assert type(mlpModel.xTest) is np.ndarray
    assert type(mlpModel.yTrain) is np.ndarray
    assert type(mlpModel.yTest) is np.ndarray
    
def test_initialValues():
    # import dataset from example
    df = pd.read_csv('tests/test-data/data-example.csv', sep=';')
    
    # get important variables
    uniqueStations = np.unique(df['station'])
    uniqueYears = np.unique(df['year'])
    varListInputs = ['tx', 'tn', 'rs']
    varListOutputs = ['et0']
    
    # Daset split by year
    xTrain, xTest, yTrain, yTest, scaler = splitDataByYear(
        df=df,
        station=uniqueStations[-1], 
        yearTestStart=uniqueYears[-1], 
        varListInputs=varListInputs, 
        varListOutputs=varListOutputs)
    
    mlpModel = MultiLayerPerceptron(xTrain, xTest, yTrain, yTest)
    
    assert mlpModel.nInputs == len(varListInputs)
    assert mlpModel.nOutputs == len(varListOutputs)
    assert mlpModel.lagDays == 1
    assert mlpModel.batchTrain ==  xTrain.shape[0]
    assert mlpModel.batchTrain ==  yTrain.shape[0]
    assert mlpModel.batchTest ==  xTest.shape[0]
    assert mlpModel.batchTest ==  yTest.shape[0]

    # dataset split by station
    xTrain, xTest, yTrain, yTest, scaler = splitDataByStation(
        df=df,
        stationsTrainList=uniqueStations[:-1], 
        stationTest=uniqueStations[-1], 
        varListInputs=varListInputs, 
        varListOutputs=varListOutputs)
    
    mlpModel = MultiLayerPerceptron(xTrain, xTest, yTrain, yTest)
    
    assert mlpModel.nInputs == len(varListInputs)
    assert mlpModel.nOutputs == len(varListOutputs)
    assert mlpModel.lagDays == 1
    assert mlpModel.batchTrain ==  xTrain.shape[0]
    assert mlpModel.batchTrain ==  yTrain.shape[0]
    assert mlpModel.batchTest ==  xTest.shape[0]
    assert mlpModel.batchTest ==  yTest.shape[0]
    
def test_buildRandomModel():
    
    # import dataset from example
    df = pd.read_csv('tests/test-data/data-example.csv', sep=';')
    
    # get important variables
    uniqueStations = np.unique(df['station'])
    uniqueYears = np.unique(df['year'])
    varListInputs = ['tx', 'tn', 'rs']
    varListOutputs = ['et0']
    
    # Daset split by year
    xTrain, xTest, yTrain, yTest, scaler = splitDataByYear(
        df=df,
        station=uniqueStations[-1], 
        yearTestStart=uniqueYears[-1], 
        varListInputs=varListInputs, 
        varListOutputs=varListOutputs)
    
    mlpModel = MultiLayerPerceptron(xTrain, xTest, yTrain, yTest)
    
    try:
        compiledModel = mlpModel.buildModel(
            hiddenLayers = 4,
            neurons = 10,
            activation = 'relu',
            optimizer = 'adam')
    except:
        print('The model could not be built')
        raise
    
def test_checkPredictionDimensionsSingleTimeStep():
    # import dataset from example
    df = pd.read_csv('tests/test-data/data-example.csv', sep=';')
    
    # get important variables
    uniqueStations = np.unique(df['station'])
    uniqueYears = np.unique(df['year'])
    varListInputs = ['tx', 'tn', 'rs']
    varListOutputs = ['et0']
    
    # Daset split by year
    xTrain, xTest, yTrain, yTest, scaler = splitDataByYear(
        df=df,
        station=uniqueStations[-1], 
        yearTestStart=uniqueYears[-1], 
        varListInputs=varListInputs, 
        varListOutputs=varListOutputs)
    
    mlpModel = MultiLayerPerceptron(xTrain, xTest, yTrain, yTest)
    
    compiledModel = mlpModel.buildModel(
            hiddenLayers = 4,
            neurons = 10,
            activation = 'relu',
            optimizer = 'adam')

    mlpModel.trainFullTrainingData(compiledModel)
    yPred = mlpModel.predictFullTestingData(compiledModel)
    print(yPred.shape)
    print(yTest.shape)
    
    assert yPred.shape == yTest.shape

def test_saveAndLoadModel():
    # import dataset from example
    df = pd.read_csv('tests/test-data/data-example.csv', sep=';')
    
    # get important variables
    uniqueStations = np.unique(df['station'])
    uniqueYears = np.unique(df['year'])
    varListInputs = ['tx', 'tn', 'rs']
    varListOutputs = ['et0']
    
    # Daset split by year
    xTrain, xTest, yTrain, yTest, scaler = splitDataByYear(
        df=df,
        station=uniqueStations[-1], 
        yearTestStart=uniqueYears[-1], 
        varListInputs=varListInputs, 
        varListOutputs=varListOutputs)
    
    mlpModel = MultiLayerPerceptron(xTrain, xTest, yTrain, yTest)
    
    compiledModel = mlpModel.buildModel(
            hiddenLayers = 4,
            neurons = 10,
            activation = 'relu',
            optimizer = 'adam')

    trainedModel = mlpModel.trainFullTrainingData(compiledModel)
    mlpModel.saveModel(
        model = trainedModel,
        fileName = 'tests/test-models/mlpModelTest')
    
    assert os.path.exists('tests/test-models/mlpModelTest.h5'), 'The model was not saved'

    loadedModel = mlpModel.loadModel(fileName = 'tests/test-models/mlpModelTest.h5')
    yPred = mlpModel.predictFullTestingData(loadedModel)
    
    assert yPred.shape == yTest.shape, 'The model was not successfully loaded'
    
def test_checkPredictionDimensionsMultipleTimeStep():
    # import dataset from example
    df = pd.read_csv('tests/test-data/data-example.csv', sep=';')
    
    # get important variables
    uniqueStations = np.unique(df['station'])
    trainLength = 0.7
    lagTimeSteps = 10
    forecastTimeSteps = 4
    varListInputs = ['tx', 'tn', 'rs']
    varOutput = 'et0'
    
    # Daset split by year
    xTrain, xTest, yTrain, yTest, scaler = splitDataByPercentageWithLagDays(
        df, 
        station = uniqueStations[0],
        trainLength = trainLength,
        lagTimeSteps = lagTimeSteps,
        forecastTimeSteps = forecastTimeSteps,
        varListInputs = varListInputs, 
        varOutput = 'et0')
    
    mlpModel = MultiLayerPerceptron(xTrain, xTest, yTrain, yTest)
    
    compiledModel = mlpModel.buildModel(
            hiddenLayers = 4,
            neurons = 10,
            activation = 'relu',
            optimizer = 'adam')

    mlpModel.trainFullTrainingData(compiledModel)
    yPred = mlpModel.predictFullTestingData(compiledModel)
    print(yPred.shape)
    print(yTest.shape)
    
    assert yPred.shape == yTest.shape
    
def test_checkModelIsLearning():
    # import dataset from example
    df = pd.read_csv('tests/test-data/data-example.csv', sep=';')
    
    # get important variables
    uniqueStations = np.unique(df['station'])
    uniqueYears = np.unique(df['year'])
    varListInputs = ['tx', 'tn', 'rs']
    varListOutputs = ['et0']
    
    # Daset split by year
    xTrain, xTest, yTrain, yTest, scaler = splitDataByYear(
        df=df,
        station=uniqueStations[-1], 
        yearTestStart=uniqueYears[-1], 
        varListInputs=varListInputs, 
        varListOutputs=varListOutputs)
    
    mlpModel = MultiLayerPerceptron(xTrain, xTest, yTrain, yTest)
    
    compiledModel = mlpModel.buildModel(
            hiddenLayers = 2,
            neurons = 5,
            activation = 'relu',
            optimizer = 'adam')

    mlpModel.trainFullTrainingData(compiledModel, epochs=100, showGraph=False)
    yPred = mlpModel.predictFullTestingData(compiledModel)
    
    mbe = getMeanBiasError(yTest, yPred)
    rmse = getRootMeanSquaredError(yTest, yPred)
    
    assert True
    
def test_checkBayesianOptimizationHoldout():
    # import dataset from example
    df = pd.read_csv('tests/test-data/data-example.csv', sep=';')
    
    # get important variables
    uniqueStations = np.unique(df['station'])
    uniqueYears = np.unique(df['year'])
    varListInputs = ['tx', 'tn', 'rs']
    varListOutputs = ['et0']
    
    # Daset split by year
    xTrain, xTest, yTrain, yTest, scaler = splitDataByYear(
        df=df,
        station=uniqueStations[-1], 
        yearTestStart=uniqueYears[-1], 
        varListInputs=varListInputs, 
        varListOutputs=varListOutputs)
    
    mlpModel = MultiLayerPerceptron(xTrain, xTest, yTrain, yTest)
    
    # holdout without shuffle
    mlpModelBayes, bestParams = mlpModel.bayesianOptimization(
        hiddenLayersList=[1,2], 
        neuronsList=[5, 10], 
        activationList=['relu'], 
        optimizerList=['adam'], 
        epochsList=[50,100], 
        bayesianEpochs=5, 
        randomStart=4, 
        validationSplit=0.2, 
        shuffle=False)
    mlpModel.trainFullTrainingData(mlpModelBayes, epochs=bestParams['epochs'], showGraph=False)
    yPred = mlpModel.predictFullTestingData(mlpModelBayes)
    mbe = getMeanBiasError(yTest, yPred)
    rmse = getRootMeanSquaredError(yTest, yPred)
    
    
    # holdout without shuffle
    mlpModelBayes, bestParams = mlpModel.bayesianOptimization(
        hiddenLayersList=[1,2], 
        neuronsList=[5, 10], 
        activationList=['relu'], 
        optimizerList=['adam'], 
        epochsList=[50,100], 
        bayesianEpochs=5, 
        randomStart=4, 
        validationSplit=0.2, 
        shuffle=True)
    mlpModel.trainFullTrainingData(mlpModelBayes, epochs=bestParams['epochs'], showGraph=False)
    yPred = mlpModel.predictFullTestingData(mlpModelBayes)
    mbe = getMeanBiasError(yTest, yPred)
    rmse = getRootMeanSquaredError(yTest, yPred)
    
    
    
    # holdout without shuffle
    mlpModelBayes, bestParams = mlpModel.bayesianOptimization(
        hiddenLayersList=[1,2], 
        neuronsList=[5, 10], 
        activationList=['relu'], 
        optimizerList=['adam'], 
        epochsList=[50,100], 
        bayesianEpochs=5, 
        randomStart=4, 
        nFolds=4,
        crossVal=True)
    mlpModel.trainFullTrainingData(mlpModelBayes, epochs=bestParams['epochs'], showGraph=False)
    yPred = mlpModel.predictFullTestingData(mlpModelBayes)
    mbe = getMeanBiasError(yTest, yPred)
    rmse = getRootMeanSquaredError(yTest, yPred)
    
    
    assert True
    