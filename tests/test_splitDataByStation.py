#%%
import os

import pytest
import pandas as pd
import numpy as np
from agroml.utils.splitDataByStation import splitDataByStation

def test_dimensionsFromTrainAndTest():
    df = pd.read_csv('tests/test-data/data-example.csv', sep=';')
    uniqueStations = np.unique(df['station'])

    stationsTrainList=uniqueStations[:-1]
    stationTest=uniqueStations[-1]
    varListInputs=['tx', 'tn', 'rs']
    varListOutputs=[ 'et0']

    xTrain, xTest, yTrain, yTest, scaler = splitDataByStation(
        df=df,
        stationsTrainList=stationsTrainList, 
        stationTest=stationTest, 
        varListInputs=varListInputs, 
        varListOutputs=varListOutputs
    )

    # check the number of stations
    lenDataTrainStations = df[df['station'].isin(stationsTrainList)].shape[0]
    lenDataTestStations = df[df['station']==stationTest].shape[0]

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

    stationsTrainList=uniqueStations[:-1]
    stationTest=uniqueStations[-1]
    varListInputs=['tx', 'tn', 'rs']
    varListOutputs=[ 'et0']

    xTrain, xTest, yTrain, yTest, scaler = splitDataByStation(
        df=df,
        stationsTrainList=stationsTrainList, 
        stationTest=stationTest, 
        varListInputs=varListInputs, 
        varListOutputs=varListOutputs
    )
    xTrain2, xTest2, yTrain2, yTest2, scaler = splitDataByStation(
        df=df,
        stationsTrainList=stationsTrainList, 
        stationTest=stationTest, 
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

    stationsTrainList=uniqueStations[:-1]
    stationTest=uniqueStations[-1]
    varListInputs=['tx', 'tn', 'rs']
    varListOutputs=[ 'et0']

    xTrain, xTest, yTrain, yTest, scaler = splitDataByStation(
        df=df,
        stationsTrainList=stationsTrainList, 
        stationTest=stationTest, 
        varListInputs=varListInputs, 
        varListOutputs=varListOutputs,
        preprocessing='normalization'
    )
    
    assert np.max(xTrain) == 1.0
# %%
