import pandas as pd
import numpy as np
from icecream import ic

from sklearn.preprocessing import StandardScaler, MinMaxScaler

def detectMissingDates(df, idx_start, idx_end):
    """ It detects if there is any missing date between the edge dates

    Arguments:
        df (Dataframe) -> Dataframe containing the database data
        idx_start (int) -> Row index indicating the initial date
        idx_end (int) -> Row index indicating the final date

    Output:
        out (int) - It returns the len of missing dates in this period
    
    """
    completeDatesRange = pd.date_range(df['date'][idx_start], df['date'][idx_end])
    currentDates = df['date'][idx_start:idx_end+1]
    missing_dates = completeDatesRange.difference(currentDates)

    return len(missing_dates)

def splitDataByPercentageWithLagDays(
    df, 
    station,
    trainLength,
    lagTimeSteps,
    forecastTimeSteps,
    varListInputs, 
    varOutput,
    preprocessing = 'standardization'):

    """
    It splits the dataset into training and testing according to the station.
    Very useful in regional scenarios

    Inputs:
        df (dataframe) - Input DataFrame
        
        station  (str) - station
        
        trainLength (float) -  It represents the percentage of training data
        
        lagTimeSteps (int) - number of timestamps to use from the past as input
        
        forecastTimeSteps (int) - Number of timesteps to forecast from future
        
        varListInputs (list) - List with input variable configuration
        
        varOutput (str) - String with target variable
        
        preprocessing (str) - 'Standardization' or 'Normalization' or 'None'

    outputs:
        xTrain (np.array) - shape(batchTrain, lagTimeSteps, nFeaturesInput)
        
        xTest (np.array) - shape(batchTest, lagTimeSteps, nFeaturesInput)
        
        yTrain (np.array) - shape(batchTrain, nFeaturesOutput)
        
        yTest (np.array) - shape(batchTest, nFeaturesOutput)
    """
    # errors
    assert 'station' in df.columns, "'station'does not exist in the dataframe"
    assert 'date' in df.columns, "'date' does not exist in the dataframe"

    # join all var configurations
    varList = varListInputs.copy()
    varList.append(varOutput)

    # filter data by station
    df = df[df['station']==station]
    df['date'] = pd.to_datetime(df['date'])
    df.reset_index(drop=True, inplace=True)
    
    # get index train
    idxTrain = int(trainLength*df.shape[0])
    
    # split to input and target
    X = df.loc[:, varListInputs]
    y = df.loc[:, varOutput]
    
    # normalization / standardization
    if preprocessing == 'standardization':
        scaler = StandardScaler()
        scaler.fit(X.iloc[:idxTrain])
        X = pd.DataFrame(scaler.transform(X),columns=X.columns)
        
    elif preprocessing == 'normalization':
        scaler = MinMaxScaler()
        scaler.fit(X.iloc[:idxTrain])
        X = pd.DataFrame(scaler.transform(X),columns=X.columns)
    
    
    input_sequence, target_sequence, season_list = list(), list(), list()
    for i in range(df.shape[0]-(lagTimeSteps+forecastTimeSteps)):
        #Detect there is no gap date
        missing_dates = detectMissingDates(df,i, i+lagTimeSteps+forecastTimeSteps)

        if missing_dates == 0:
            indexIn = [j for j in range(i, i + lagTimeSteps)]
            indexTar = [j for j in range(i + lagTimeSteps, i + lagTimeSteps + forecastTimeSteps)]

            input_sequence.append(X.iloc[indexIn].values)
            target_sequence.append(y.iloc[indexTar].values)
    
    X = np.array(input_sequence).astype('float32')
    y = np.array(target_sequence).astype('float32')
    
    xTrain = X[:idxTrain] 
    xTest= X[idxTrain:] 
    yTrain = y[:idxTrain] 
    yTest= y[idxTrain:]
   
    return xTrain, xTest, yTrain, yTest, scaler