import pandas as pd
from icecream import ic

from agroml.utils.splitDataByYear import splitDataByYear
from agroml.models.svmRegression import SupportVectorMachine
from agroml.models.rfRegression import RandomForest
from agroml.models.elmRegression import ExtremeLearningMachine
from agroml.models.mlpRegression import MultiLayerPerceptron
from agroml.models.cnnRegression import ConvolutionalNeuralNetwork
from agroml.models.lstmRegression import LongShortTermMemory
from agroml.models.transformerCnnRegression import transformerCNN
from agroml.models.transformerLstmRegression import transformerLSTM


from agroml.utils.statistics import *
from agroml.utils.plots import *

# import dataset from example
df = pd.read_csv('tests/test-data/data-example.csv', sep=';')

# get important variables
uniqueStations = np.unique(df['station'])
uniqueYears = np.unique(df['year'])
varListInputs = ['tx', 'tn', 'rs', 'day']
varListOutputs = ['et0']

# split data to train and test datasets
xTrain, xTest, yTrain, yTest, scaler = splitDataByYear(
    df=df,
    station=uniqueStations[-1], 
    yearTestStart=uniqueYears[-3], 
    varListInputs=varListInputs, 
    varListOutputs=varListOutputs,
    preprocessing = 'standardization')


##################################################
# MLP
##################################################
# create model
mlModel = MultiLayerPerceptron(xTrain, xTest, yTrain, yTest)
# Tuning optimization
mlModelBayes, bestParams = mlModel.bayesianOptimization(
    hiddenLayersList=[1,2], 
    neuronsList=[1, 20], 
    activationList=['relu'], 
    optimizerList=['adam'], 
    epochsList=[50,100], 
    bayesianEpochs=5, 
    randomStart=4, 
    validationSplit=0.2, 
    shuffle=False)

# train best model with the full dataset
mlModel.trainFullTrainingData(mlModelBayes, showGraph=False)
yPred = mlModel.predictFullTestingData(mlModelBayes)
mbe = getMeanBiasError(yTest, yPred)
rmse = getRootMeanSquaredError(yTest, yPred)
nse = getNashSuteliffeEfficiency(yTest, yPred)

# plot predictions vs. measured
plotGraphLinealRegresion(
    x = yTest, 
    xName = 'Measures values', 
    y = yPred, 
    yName = 'Predicted values')

##################################################
# SVM
##################################################
mlModel = SupportVectorMachine(xTrain, xTest, yTrain, yTest)
mlModelBayes, bestParams = mlModel.bayesianOptimization(
    kernelList = ['linear', 'poly', 'rbf', 'sigmoid'], 
    cList = [0.01, 1], 
    epsilonList = [0.01, 1],
    bayesianEpochs=5, 
    randomStart=4, 
    validationSplit=0.2, 
    shuffle=False)

# train best model with the full dataset
mlModel.trainFullTrainingData(mlModelBayes, showGraph=False)
yPred = mlModel.predictFullTestingData(mlModelBayes)
mbe = getMeanBiasError(yTest, yPred)
rmse = getRootMeanSquaredError(yTest, yPred)
nse = getNashSuteliffeEfficiency(yTest, yPred)

# plot predictions vs. measured
plotGraphLinealRegresion(
    x = yTest, 
    xName = 'Measures values', 
    y = yPred, 
    yName = 'Predicted values')

##################################################
# ELM
##################################################
mlModel = ExtremeLearningMachine(xTrain, xTest, yTrain, yTest)
mlModelBayes, bestParams = mlModel.bayesianOptimization(
    neuronsList = [10, 150], 
    activationList = ['lin','sigm','tanh','rbf_l1','rbf_l2','rbf_linf'],
    bayesianEpochs=5, 
    randomStart=4, 
    validationSplit=0.2, 
    shuffle=False)

# train best model with the full dataset
mlModel.trainFullTrainingData(mlModelBayes, showGraph=False)
yPred = mlModel.predictFullTestingData(mlModelBayes)
mbe = getMeanBiasError(yTest, yPred)
rmse = getRootMeanSquaredError(yTest, yPred)
nse = getNashSuteliffeEfficiency(yTest, yPred)

# plot predictions vs. measured
plotGraphLinealRegresion(
    x = yTest, 
    xName = 'Measures values', 
    y = yPred, 
    yName = 'Predicted values')


##################################################
# RF
##################################################
mlModel = RandomForest(xTrain, xTest, yTrain, yTest)
mlModelBayes, bestParams = mlModel.bayesianOptimization(
    bayesianEpochs=5, 
    randomStart=4, 
    validationSplit=0.2, 
    shuffle=False)

# train best model with the full dataset
mlModel.trainFullTrainingData(mlModelBayes, showGraph=False)
yPred = mlModel.predictFullTestingData(mlModelBayes)
mbe = getMeanBiasError(yTest, yPred)
rmse = getRootMeanSquaredError(yTest, yPred)
nse = getNashSuteliffeEfficiency(yTest, yPred)

# plot predictions vs. measured
plotGraphLinealRegresion(
    x = yTest, 
    xName = 'Measures values', 
    y = yPred, 
    yName = 'Predicted values')

##################################################
# CNN
##################################################
mlModel = ConvolutionalNeuralNetwork(xTrain, xTest, yTrain, yTest)
mlModelBayes, bestParams = mlModel.bayesianOptimization(
    bayesianEpochs=5, 
    randomStart=4, 
    validationSplit=0.2, 
    shuffle=False)

# train best model with the full dataset
mlModel.trainFullTrainingData(mlModelBayes, showGraph=False)
yPred = mlModel.predictFullTestingData(mlModelBayes)
mbe = getMeanBiasError(yTest, yPred)
rmse = getRootMeanSquaredError(yTest, yPred)
nse = getNashSuteliffeEfficiency(yTest, yPred)

# plot predictions vs. measured
plotGraphLinealRegresion(
    x = yTest, 
    xName = 'Measures values', 
    y = yPred, 
    yName = 'Predicted values')

##################################################
# LSTM
##################################################
mlModel = LongShortTermMemory(xTrain, xTest, yTrain, yTest)
mlModelBayes, bestParams = mlModel.bayesianOptimization(
    bayesianEpochs=5, 
    randomStart=4, 
    validationSplit=0.2, 
    shuffle=False)

# train best model with the full dataset
mlModel.trainFullTrainingData(mlModelBayes, showGraph=False)
yPred = mlModel.predictFullTestingData(mlModelBayes)
mbe = getMeanBiasError(yTest, yPred)
rmse = getRootMeanSquaredError(yTest, yPred)
nse = getNashSuteliffeEfficiency(yTest, yPred)

# plot predictions vs. measured
plotGraphLinealRegresion(
    x = yTest, 
    xName = 'Measures values', 
    y = yPred, 
    yName = 'Predicted values')

##################################################
# TransformerCNN
##################################################
mlModel = transformerCNN(xTrain, xTest, yTrain, yTest)
mlModelBayes, bestParams = mlModel.bayesianOptimization(
    bayesianEpochs=5, 
    randomStart=4, 
    validationSplit=0.2, 
    shuffle=False)

# train best model with the full dataset
mlModel.trainFullTrainingData(mlModelBayes, showGraph=False)
yPred = mlModel.predictFullTestingData(mlModelBayes)
mbe = getMeanBiasError(yTest, yPred)
rmse = getRootMeanSquaredError(yTest, yPred)
nse = getNashSuteliffeEfficiency(yTest, yPred)

# plot predictions vs. measured
plotGraphLinealRegresion(
    x = yTest, 
    xName = 'Measures values', 
    y = yPred, 
    yName = 'Predicted values')

##################################################
# TransformerLSTM
##################################################
mlModel = transformerLSTM(xTrain, xTest, yTrain, yTest)
mlModelBayes, bestParams = mlModel.bayesianOptimization(
    bayesianEpochs=5, 
    randomStart=4, 
    validationSplit=0.2, 
    shuffle=False)

# train best model with the full dataset
mlModel.trainFullTrainingData(mlModelBayes, showGraph=False)
yPred = mlModel.predictFullTestingData(mlModelBayes)
mbe = getMeanBiasError(yTest, yPred)
rmse = getRootMeanSquaredError(yTest, yPred)
nse = getNashSuteliffeEfficiency(yTest, yPred)

# plot predictions vs. measured
plotGraphLinealRegresion(
    x = yTest, 
    xName = 'Measures values', 
    y = yPred, 
    yName = 'Predicted values')

