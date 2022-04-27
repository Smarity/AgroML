
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def splitDataByStation( df, stationsTrainList, stationTest, varListInputs, varListOutputs, preprocessing = 'standardization'):
    """
    It splits the dataset into training and testing according to the station.
    Very useful in regional scenarios.

    Note: the dataframe must have a column names 
        'station' (str) 
        'date (datetime)

    Inputs:
        df (dataframe) - Input DataFrame
        stationsTrainList  (list) - List of stations for training
        stationTest (str) - String with the station name to test.
        varListInputs (list) - List with input variable configuration
        varListOutputs (list) - List with target variables
        preprocessing (str) - 'Standardization' or 'Normalization' or 'None'

    outputs: 
        xTrain (np.array) - shape(batchTrain, 1, nFeaturesInput)
        xTest (np.array) - shape(batchTrain, 1, nFeaturesInput)
        yTrain (np.array) - shape(batchTest, nFeaturesOutput)
        yTest (np.array) - shape(batchTest, nFeaturesOutput)
    """
    # errors
    assert 'station' in df.columns, "'station'does not exist in the dataframe"
    assert 'date' in df.columns, "'date' does not exist in the dataframe"

    # join all var configurations
    varList = varListInputs + varListOutputs

    # split to train and test
    dfStationTrain = df[df['station'].isin(stationsTrainList)]
    dfStationTrain = dfStationTrain.filter(items=varList)
    xTrain = dfStationTrain.filter(items=varListInputs).to_numpy()
    yTrain = dfStationTrain.filter(items=varListOutputs).to_numpy()

    dfStationTest = df[df['station']==stationTest]
    dfStationTest = dfStationTest.filter(items=varList)
    xTest = dfStationTest.filter(items=varListInputs).to_numpy()
    yTest = dfStationTest.filter(items=varListOutputs).to_numpy()

    # standardization or normalization
    if preprocessing == 'standardization':
        scaler = StandardScaler()
        scaler.fit(xTrain)
        xTrain = scaler.transform(xTrain)
        xTest = scaler.transform(xTest)

    elif preprocessing == 'normalization':
        scaler = MinMaxScaler()
        scaler.fit(xTrain)
        xTrain = scaler.transform(xTrain)
        xTest = scaler.transform(xTest)
    else:
        scaler ='none'

    xTrain = xTrain.reshape(len(dfStationTrain), 1, len(varListInputs))
    xTest = xTest.reshape(len(dfStationTest), 1, len(varListInputs))
    yTrain = yTrain.reshape(len(dfStationTrain), len(varListOutputs))
    yTest = yTest.reshape(len(dfStationTest), len(varListOutputs))
    
    return xTrain, xTest, yTrain, yTest, scaler

