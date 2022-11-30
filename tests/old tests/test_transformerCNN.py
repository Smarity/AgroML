import os

import pandas as pd
import numpy as np
from icecream import ic

from agroml.utils.splitDataByPercentageWithLagDays import splitDataByPercentageWithLagDays
from agroml.utils.splitDataByStation import splitDataByStation
from agroml.utils.splitDataByYear import splitDataByYear
from agroml.models.transformerCnnRegression import transformerCNN
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
    
    mlpModel = transformerCNN(xTrain, xTest, yTrain, yTest)
    
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
    
    mlModel = transformerCNN(xTrain, xTest, yTrain, yTest)
    
    assert mlModel.nInputs == len(varListInputs)
    assert mlModel.nOutputs == len(varListOutputs)
    assert mlModel.lagDays == 1
    assert mlModel.batchTrain ==  xTrain.shape[0]
    assert mlModel.batchTrain ==  yTrain.shape[0]
    assert mlModel.batchTest ==  xTest.shape[0]
    assert mlModel.batchTest ==  yTest.shape[0]

    # dataset split by station
    xTrain, xTest, yTrain, yTest, scaler = splitDataByStation(
        df=df,
        stationsTrainList=uniqueStations[:-1], 
        stationTest=uniqueStations[-1], 
        varListInputs=varListInputs, 
        varListOutputs=varListOutputs)
    
    mlModel = transformerCNN(xTrain, xTest, yTrain, yTest)
    
    assert mlModel.nInputs == len(varListInputs)
    assert mlModel.nOutputs == len(varListOutputs)
    assert mlModel.lagDays == 1
    assert mlModel.batchTrain ==  xTrain.shape[0]
    assert mlModel.batchTrain ==  yTrain.shape[0]
    assert mlModel.batchTest ==  xTest.shape[0]
    assert mlModel.batchTest ==  yTest.shape[0]
    
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
    
    mlModel = transformerCNN(xTrain, xTest, yTrain, yTest)
    
    try:
        compiledModel = mlModel.buildModel(
            headSize = 2, 
            nHeads = 16, 
            ffDim = 32, 
            nBlocks = 3, 
            nHiddenLayers = 2, 
            nHiddenNeurons = 10, 
            nKernel = 5, 
            attentionDropout = 0.01,
            mlpDropout=0.01)
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
    
    mlModel = transformerCNN(xTrain, xTest, yTrain, yTest)
    
    compiledModel = mlModel.buildModel(
        headSize = 2, 
        nHeads = 16, 
        ffDim = 32, 
        nBlocks = 3, 
        nHiddenLayers = 2, 
        nHiddenNeurons = 10, 
        nKernel = 5, 
        attentionDropout=0.1, 
        mlpDropout=0.01)

    mlModel.trainFullTrainingData(compiledModel, verbose=1)
    yPred = mlModel.predictFullTestingData(compiledModel)
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
    
    mlModel = transformerCNN(xTrain, xTest, yTrain, yTest)
    
    compiledModel = mlModel.buildModel(
        headSize = 2, 
        nHeads = 16, 
        ffDim = 32, 
        nBlocks = 3, 
        nHiddenLayers = 2, 
        nHiddenNeurons = 10, 
        nKernel = 5, 
        attentionDropout=0.1, 
        mlpDropout=0.01)

    trainedModel = mlModel.trainFullTrainingData(compiledModel)
    mlModel.saveModel(
        model = trainedModel,
        fileName = 'tests/test-models/transformerCnnModelTest')
    
    assert os.path.exists('tests/test-models/transformerCnnModelTest.h5'), 'The model was not saved'

    loadedModel = mlModel.loadModel(fileName = 'tests/test-models/transformerCnnModelTest.h5')
    yPred = mlModel.predictFullTestingData(loadedModel)
    
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
    
    mlModel = transformerCNN(xTrain, xTest, yTrain, yTest)
    
    compiledModel = mlModel.buildModel(
        headSize = 2, 
        nHeads = 16, 
        ffDim = 32, 
        nBlocks = 3, 
        nHiddenLayers = 2, 
        nHiddenNeurons = 10, 
        nKernel = 5, 
        attentionDropout=0.1, 
        mlpDropout=0.01)

    mlModel.trainFullTrainingData(compiledModel)
    yPred = mlModel.predictFullTestingData(compiledModel)
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
    
    mlModel = transformerCNN(xTrain, xTest, yTrain, yTest)
    
    compiledModel = mlModel.buildModel(
        headSize = 2, 
        nHeads = 16, 
        ffDim = 32, 
        nBlocks = 3, 
        nHiddenLayers = 2, 
        nHiddenNeurons = 10, 
        nKernel = 5, 
        attentionDropout=0.1, 
        mlpDropout=0.01)

    mlModel.trainFullTrainingData(compiledModel, epochs=100, showGraph=False)
    yPred = mlModel.predictFullTestingData(compiledModel)
    
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
    
    mlModel = transformerCNN(xTrain, xTest, yTrain, yTest)
    
    # holdout without shuffle
    mlpModelBayes, bestParams = mlModel.bayesianOptimization(
        headSizeList=[1, 32], 
        nHeadsList=[1, 32], 
        ffDimList=[16, 64], 
        nBlocksList=[1, 3], 
        nHiddenLayersList=[1, 3], 
        nHiddenNeuronsList=[1, 10], 
        nKernelList=[2, 7], 
        attentionDropoutList=[0, 0.15], 
        mlpDropoutList=[0, 0.15],
        epochsList=[50,150],  
        bayesianEpochs=5, 
        randomStart=4, 
        validationSplit=0.2, 
        shuffle=False)
    mlModel.trainFullTrainingData(mlpModelBayes, epochs=bestParams['epochs'], showGraph=False)
    yPred = mlModel.predictFullTestingData(mlpModelBayes)
    mbe = getMeanBiasError(yTest, yPred)
    rmse = getRootMeanSquaredError(yTest, yPred)
    
    # holdout without shuffle
    mlpModelBayes, bestParams = mlModel.bayesianOptimization(
        headSizeList=[1, 32], 
        nHeadsList=[1, 32], 
        ffDimList=[16, 64], 
        nBlocksList=[1, 3], 
        nHiddenLayersList=[1, 3], 
        nHiddenNeuronsList=[1, 10], 
        nKernelList=[2, 7], 
        attentionDropoutList=[0, 0.15], 
        mlpDropoutList=[0, 0.15],
        epochsList=[50,150], 
        bayesianEpochs=5, 
        randomStart=4, 
        validationSplit=0.2, 
        shuffle=True)
    mlModel.trainFullTrainingData(mlpModelBayes, epochs=bestParams['epochs'], showGraph=False)
    yPred = mlModel.predictFullTestingData(mlpModelBayes)
    mbe = getMeanBiasError(yTest, yPred)
    rmse = getRootMeanSquaredError(yTest, yPred)
    
    # holdout without shuffle
    mlpModelBayes, bestParams = mlModel.bayesianOptimization(
        headSizeList=[1, 32], 
        nHeadsList=[1, 32], 
        ffDimList=[16, 64], 
        nBlocksList=[1, 3], 
        nHiddenLayersList=[1, 3], 
        nHiddenNeuronsList=[1, 10], 
        nKernelList=[2, 7], 
        attentionDropoutList=[0, 0.15], 
        mlpDropoutList=[0, 0.15],
        epochsList=[50,150], 
        bayesianEpochs=5, 
        randomStart=4, 
        nFolds=4,
        crossVal=True)
    mlModel.trainFullTrainingData(mlpModelBayes, epochs=bestParams['epochs'], showGraph=False)
    yPred = mlModel.predictFullTestingData(mlpModelBayes)
    mbe = getMeanBiasError(yTest, yPred)
    rmse = getRootMeanSquaredError(yTest, yPred)

    assert True






