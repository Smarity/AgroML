import math
import numpy as np
import statistics
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

sqrt = math.sqrt
mean = statistics.mean

def getMeanBiasError(measValues, predValues):
    '''
    It calculates the mean bias error function (MBE).

    Arguments:
        measValues {array} - shape(batch, nOutputs)
            array with the measured values.
        predValues {array} - shape(batch, nOutputs)
            array with the prediction values

    Output:
        mbe {list}
    '''
    # convert the inputs in numpy array
    measValues = np.array(measValues).astype(float)
    predValues = np.array(predValues).astype(float)
    
    #check lengths
    assert measValues.shape==predValues.shape, 'The shapes are different'

    nOutputs = measValues.shape[1]
    
    mbeList = list()
    for i in range(nOutputs):
        delta = predValues[:, i] - measValues[:, i]
        mbe = float(sum(delta)*1.0/len(measValues))
        mbeList.append(mbe)
        
    return np.array(mbeList)
    
def getRootMeanSquaredError(measValues, predValues):

    '''
    It calculates the root mean square error (RMSE).

    Arguments:
        measValues {array} - shape(batch, nOutputs)
            array with the measured values.
        predValues {array} - shape(batch, nOutputs)
            array with the prediction values

    Output:
        rmse {list}
    '''
    # convert the inputs in numpy array
    measValues = np.array(measValues).astype(float)
    predValues = np.array(predValues).astype(float)
    
    #check lengths
    assert measValues.shape==predValues.shape, 'The shapes are different'
    
    nOutputs = measValues.shape[1]
    
    rmseList = list()
    for i in range(nOutputs):
        delta = (predValues[:, i] - measValues[:, i])**2
        rmse = sqrt(sum(delta)*1.0/len(measValues))
        rmseList.append(rmse)
    
    return np.array(rmseList)


def getMeanAbsoluteError(measValues, predValues):
    '''
    It calculates the mean abosulte error function (MBE).

    Arguments:
        measValues {array} - shape(batch, nOutputs)
            array with the measured values.
        predValues {array} - shape(batch, nOutputs)
            array with the prediction values

    Output:
        mbe {list}
    '''
    # convert the inputs in numpy array
    measValues = np.array(measValues).astype(float)
    predValues = np.array(predValues).astype(float)
    
    #check lengths
    assert measValues.shape==predValues.shape, 'The shapes are different'
    
    nOutputs = measValues.shape[1]
    
    maeList = list()
    for i in range(nOutputs):
        delta = predValues[:, i] - measValues[:, i]
        maeList.append(np.abs(delta)/predValues.shape[0])
        
    return np.array(maeList)


def getCoefficientOfDetermination(measValues, predValues):
    '''
    It calculates the coefficient of determination.

    Arguments:
        measValues {array} - shape(batch, nOutputs)
            array with the measured values.
        predValues {array} - shape(batch, nOutputs)
            array with the prediction values

    Output:
        r2 {list}
    '''
    # convert the inputs in numpy array
    measValues = np.array(measValues).astype(float)
    predValues = np.array(predValues).astype(float)
    
    #check lengths
    assert measValues.shape==predValues.shape, 'The shapes are different'
    
    nOutputs = measValues.shape[1]
    
    r2List = list()
    for i in range(nOutputs):
        xMean = mean(measValues)
        yMean = mean(predValues)
        deltaMeasured = measValues - xMean
        deltaMeasured_2 = (measValues - xMean)**2
        deltaPrediction = predValues - yMean
        deltaPrediction_2 = (predValues - yMean)**2
        
        r2 = (sum(deltaMeasured*deltaPrediction)/sqrt(sum(deltaMeasured_2)*sum(deltaPrediction_2)))**2
        r2List.append(r2)
        
    return np.array(r2List)


def getNashSuteliffeEfficiency(measValues, predValues):
    '''
    It calculates the Nash Suteliffe Efficiency.

    Arguments:
        measValues {array} - shape(batch, nOutputs)
            array with the measured values.
        predValues {array} - shape(batch, nOutputs)
            array with the prediction values

    Output:
        nse {list}
    '''
    # convert the inputs in numpy array
    measValues = np.array(measValues).astype(float)
    predValues = np.array(predValues).astype(float)
    
    #check lengths
    assert measValues.shape==predValues.shape, 'The shapes are different'
    
    nOutputs = measValues.shape[1]
    
    nseList = list()
    for i in range(nOutputs):
        nseList.append(r2_score(measValues, predValues))

    return nseList


def getLinearRegression(measValues, predValues):
    '''
    It calculates the lineal regression of two array of data.

    Arguments:
        measValues {array} - shape(batch, nOutputs)
            array with the measured values.
        predValues {array} - shape(batch, nOutputs)
            array with the prediction values

    Output:
        mbe {list}
    '''
    # check lengths
    assert len(measValues)==len(predValues)

    # convert the inputs in numpy array and reshape
    measValues = np.array(measValues).reshape(-1, 1)
    predValues = np.array(predValues).reshape(-1, 1)

    # Se llama a la función
    lr = LinearRegression()
    # Se buscan los coeficientes
    lr.fit(measValues, predValues)
    lrPred = np.ravel(lr.predict(measValues))
    
    return lrPred

