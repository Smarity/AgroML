
import os

import pytest
import pandas as pd
import numpy as np
from icecream import ic

from agroml.utils.splitDataByPercentageWithLagDays import splitDataByPercentageWithLagDays

def test_dimensionsFromTrainingAndTest():
    df = pd.read_csv('tests/test-data/data-example.csv', sep=';')
    uniqueStations = np.unique(df['station'])
    uniqueYears = np.unique(df['year'])
    
    trainLength = 0.7
    lagTimeSteps = 10
    forecastTimeSteps = 4
    inputList = ['tx', 'tn', 'rs']
    varOutput = 'et0'
    
    xTrain, xTest, yTrain, yTest, scaler = splitDataByPercentageWithLagDays(
        df, 
        station = 'RIAG/COR06ZZZ',
        trainLength = trainLength,
        lagTimeSteps = lagTimeSteps,
        forecastTimeSteps = forecastTimeSteps,
        varListInputs = inputList, 
        varOutput = varOutput)
    
    assert xTrain.shape[0] <= df.shape[0]*trainLength
    assert xTrain.shape[1] == lagTimeSteps
    assert xTrain.shape[2] == len(inputList)
    
    assert xTest.shape[0] <= df.shape[0]*(1-trainLength)
    assert xTest.shape[1] == lagTimeSteps
    assert xTest.shape[2] == len(inputList)
    
    assert yTrain.shape[0] == xTrain.shape[0]
    assert yTrain.shape[1] == forecastTimeSteps
    
    assert yTest.shape[0] == xTest.shape[0]
    assert yTest.shape[1] == forecastTimeSteps