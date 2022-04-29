#%%
import os

import pytest
import pandas as pd
import numpy as np
from icecream import ic
from agroml.utils.splitDataByYear import splitDataByYear

def test_dimensionsFromTrainAndTest():
    df = pd.read_csv('tests/test-data/data-example.csv', sep=';')
    uniqueStations = np.unique(df['station'])
    uniqueYears = np.unique(df['year'])

    
    varListInputs=['tx', 'tn', 'rs']
    varListOutputs=[ 'et0']
    xTrain, xTest, yTrain, yTest, scaler = splitDataByYear(
        df=df,
        station=uniqueStations[-1], 
        yearTestStart=uniqueYears[-1], 
        varListInputs=varListInputs, 
        varListOutputs=varListOutputs
    )

    dfStation = df[df['station'] == uniqueStations[-1]]
    dfStationTrain = dfStation[dfStation['year']<uniqueYears[-1]]
    lenDataTrainStations = dfStationTrain.shape[0]
    dfStationTest = dfStation[dfStation['year']>=uniqueYears[-1]]
    lenDataTestStations = dfStationTest.shape[0]

    # training stations data go to train dataset
    assert xTrain.shape[0] == lenDataTrainStations
    assert yTrain.shape[0] == lenDataTrainStations
    
    # test stations data go to test dataset
    assert xTest.shape[0] == lenDataTestStations
    assert yTest.shape[0] == lenDataTestStations

    # input variables go to x dataset
    assert xTrain.shape[2] == len(varListInputs)
    assert xTest.shape[2] == len(varListInputs)
    assert xTrain.shape[2] == xTest.shape[2]
    
    # output variables go to y dataset
    assert yTrain.shape[1] == len(varListOutputs)
    assert yTest.shape[1] == len(varListOutputs)
    assert yTrain.shape[1] == yTest.shape[1]

def test_checkStandardizationMeanAndStd():
    df = pd.read_csv('tests/test-data/data-example.csv', sep=';')
    uniqueStations = np.unique(df['station'])
    uniqueYears = np.unique(df['year'])

    
    varListInputs=['tx', 'tn', 'rs']
    varListOutputs=[ 'et0']
    xTrain, xTest, yTrain, yTest, scaler = splitDataByYear(
        df=df,
        station=uniqueStations[-1], 
        yearTestStart=uniqueYears[-1], 
        varListInputs=varListInputs, 
        varListOutputs=varListOutputs
    )
    xTrain2, xTest2, yTrain2, yTest2, scaler = splitDataByYear(
        df=df,
        station=uniqueStations[-1], 
        yearTestStart=uniqueYears[-1], 
        varListInputs=varListInputs, 
        varListOutputs=varListOutputs,
        preprocessing='standardization'
    )

    assert np.mean(xTrain) == np.mean(xTrain2)
    assert np.mean(xTest) == np.mean(xTest2)
    assert np.mean(yTrain) == np.mean(yTrain2)
    assert np.mean(yTest) == np.mean(yTest2)
    
def test_checkMinMaxScalerValues():
    df = pd.read_csv('tests/test-data/data-example.csv', sep=';')
    uniqueStations = np.unique(df['station'])
    uniqueYears = np.unique(df['year'])

    
    varListInputs=['tx', 'tn', 'rs']
    varListOutputs=[ 'et0']
    xTrain, xTest, yTrain, yTest, scaler = splitDataByYear(
        df=df,
        station=uniqueStations[-1], 
        yearTestStart=uniqueYears[-1], 
        varListInputs=varListInputs, 
        varListOutputs=varListOutputs,
        preprocessing='normalization'
    )
    
    assert np.max(xTrain) == 1.0

def test_dataDistribution():
    df = pd.read_csv('tests/test-data/data-example.csv', sep=';')
    uniqueStations = np.unique(df['station'])
    uniqueYears = np.unique(df['year'])

    
    varListInputs=['tx', 'tn', 'rs']
    varListOutputs=[ 'et0']
    xTrain, xTest, yTrain, yTest, scaler = splitDataByYear(
        df=df,
        station=uniqueStations[-1], 
        yearTestStart=uniqueYears[-1], 
        varListInputs=varListInputs, 
        varListOutputs=varListOutputs,
        preprocessing='none'
    )
    dfStation = df[df['station']==uniqueStations[-1]]
    dfStation.reset_index(drop=True, inplace=True)

    assert dfStation["tx"][0] == xTrain[0,0,0]
    assert dfStation["tn"][0] == xTrain[0,0,1]
    assert dfStation["rs"][0] == xTrain[0,0,2]
    assert dfStation["et0"][0] == yTrain[0,0]



# %%
