#%%
import os

import pytest
import pretty_errors
import pandas as pd
import numpy as np
from icecream import ic
from agroml.utils.splitDataByYear import splitDataByYear
from agroml.utils.splitDataByStation import splitDataByStation
from agroml.utils.splitDataByPercentageWithLagDays import splitDataByPercentageWithLagDays
from agroml.utils.featureSelection import selectFeatureUnivariate

def test_itWorksWithSplitByYear():
    df = pd.read_csv('tests/test-data/data-example.csv', sep=';')
    uniqueStations = np.unique(df['station'])
    uniqueYears = np.unique(df['year'])

    varListInputs=['tx', 'tn', 'tm', 'rhx', 'rhn', 'rhm' , 'rs']
    varListOutputs=[ 'et0']
    xTrain, xTest, yTrain, yTest, scaler = splitDataByYear(
        df=df,
        station=uniqueStations[-1], 
        yearTestStart=uniqueYears[-1], 
        varListInputs=varListInputs, 
        varListOutputs=varListOutputs,
    )
    xTrainNew, xTestNew, selectedFeaturesList = selectFeatureUnivariate(
        xTrain = xTrain, 
        xTest = xTest, 
        yTrain = yTrain, 
        pValue = 0.79, 
        featureList = varListInputs, 
        scoringFunction = "r_regression")
    
    assert len(selectedFeaturesList) < xTrain.shape[2]
    assert xTrainNew.shape[2] == len(selectedFeaturesList)
    assert xTestNew.shape[2] == len(selectedFeaturesList)

    xTrainNew, xTestNew, selectedFeaturesList = selectFeatureUnivariate(
        xTrain = xTrain, 
        xTest = xTest, 
        yTrain = yTrain,
        pValue = 0.9, 
        featureList = varListInputs, 
        scoringFunction = "mutual_info_regression")

    assert len(selectedFeaturesList) < xTrain.shape[2]
    assert xTrainNew.shape[2] == len(selectedFeaturesList)
    assert xTestNew.shape[2] == len(selectedFeaturesList)

def test_itWorksWithSplitByStation():
    df = pd.read_csv('tests/test-data/data-example.csv', sep=';')
    uniqueStations = np.unique(df['station'])
    uniqueYears = np.unique(df['year'])

    stationsTrainList=uniqueStations[:-1]
    stationTest=uniqueStations[-1]
    varListInputs=['tx', 'tn', 'tm', 'rhx', 'rhn', 'rhm' , 'rs']
    varListOutputs=[ 'et0']
    xTrain, xTest, yTrain, yTest, scaler = splitDataByStation(
        df=df,
        stationsTrainList=stationsTrainList, 
        stationTest=stationTest, 
        varListInputs=varListInputs, 
        varListOutputs=varListOutputs
    )
    xTrainNew, xTestNew, selectedFeaturesList = selectFeatureUnivariate(
        xTrain = xTrain, 
        xTest = xTest, 
        yTrain = yTrain, 
        pValue = 0.79, 
        featureList = varListInputs, 
        scoringFunction = "r_regression")
    
    assert len(selectedFeaturesList) < xTrain.shape[2]
    assert xTrainNew.shape[2] == len(selectedFeaturesList)
    assert xTestNew.shape[2] == len(selectedFeaturesList)

    xTrainNew, xTestNew, selectedFeaturesList = selectFeatureUnivariate(
        xTrain = xTrain, 
        xTest = xTest, 
        yTrain = yTrain,
        pValue = 0.9, 
        featureList = varListInputs, 
        scoringFunction = "mutual_info_regression")

    assert len(selectedFeaturesList) < xTrain.shape[2]
    assert xTrainNew.shape[2] == len(selectedFeaturesList)
    assert xTestNew.shape[2] == len(selectedFeaturesList)

""" NOT WORKING YET
def test_itWorksWithSplitLagDays():
    df = pd.read_csv('tests/test-data/data-example.csv', sep=';')
    uniqueStations = np.unique(df['station'])
    uniqueYears = np.unique(df['year'])

    stationsTrainList=uniqueStations[:-1]
    stationTest=uniqueStations[-1]
    varListInputs=['tx', 'tn', 'tm', 'rhx', 'rhn', 'rhm' , 'rs']
    varListOutputs=[ 'et0']
    trainLength = 0.7
    lagTimeSteps = 10
    forecastTimeSteps = 4
    
    xTrain, xTest, yTrain, yTest, scaler = splitDataByPercentageWithLagDays(
        df, 
        station = 'RIAG/COR06ZZZ',
        trainLength = trainLength,
        lagTimeSteps = lagTimeSteps,
        forecastTimeSteps = forecastTimeSteps,
        varListInputs = varListInputs, 
        varOutput = varListOutputs)

    xTrainNew, xTestNew, selectedFeaturesList = selectFeatureUnivariate(
        xTrain = xTrain, 
        xTest = xTest, 
        yTrain = yTrain, 
        pValue = 0.79, 
        featureList = varListInputs, 
        scoringFunction = "r_regression")
    
    assert len(selectedFeaturesList) < xTrain.shape[2]
    assert xTrainNew.shape[2] == len(selectedFeaturesList)
    assert xTestNew.shape[2] == len(selectedFeaturesList)

    xTrainNew, xTestNew, selectedFeaturesList = selectFeatureUnivariate(
        xTrain = xTrain, 
        xTest = xTest, 
        yTrain = yTrain,
        pValue = 0.9, 
        featureList = varListInputs, 
        scoringFunction = "mutual_info_regression")

    assert len(selectedFeaturesList) < xTrain.shape[2]
    assert xTrainNew.shape[2] == len(selectedFeaturesList)
    assert xTestNew.shape[2] == len(selectedFeaturesList)"""
