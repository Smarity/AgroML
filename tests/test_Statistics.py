
import numpy as np
from agroml.utils.statistics import *

MeasuredValues = np.array([
        [1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5]
    ])
PredictedValues = np.array([
        [0.9, 2.1, 3.3, 3.6, 4.5],
        [0.9, 2.1, 3.3, 3.6, 4.5],
        [0.9, 2.1, 3.3, 3.6, 4.5],
        [0.9, 2.1, 3.3, 3.6, 4.5],
        [0.9, 2.1, 3.3, 3.6, 4.5]
    ])

def test_mbe():
    mbeResults = getMeanBiasError(MeasuredValues, PredictedValues)
    mbeResults = mbeResults.astype('float16')
    comparison = (mbeResults == np.array([-0.1, 0.1, 0.3, -0.4, -0.5]).astype('float16'))
    assert comparison.all()
    
def test_rmse():
    rmseResults = getRootMeanSquaredError(MeasuredValues, PredictedValues)
    rmseResults = rmseResults.astype('float16')
    print(rmseResults)
    comparison = (rmseResults == np.array([0.1, 0.1, 0.3, 0.4, 0.5]).astype('float16'))
    assert comparison.all()