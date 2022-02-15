################################################################################
# In this script we can find several algorithms to select the fittest features
################################################################################

import numpy as np
import pandas as pd
from icecream import ic
from sklearn.feature_selection import SelectKBest, VarianceThreshold
from sklearn.feature_selection import f_regression, r_regression
from sklearn.feature_selection import mutual_info_regression

def selectFeaturesBasedOnVarianceThreshold(df, xTrain, xTest,featureList, pValue=0.8):
    """
    The Variance Threshold is a simple baseline approach to feature selection.
    It removes all features whose variance doesn't meet some threshold.


    For binary data, as an example, suppose that we have a dataset with boolean features, and we
    want to remove all features that are either one or zero in more than 80% of the samples.

    Var[X] = Threshold = pValue*(1-pValue) = 0.8*(1-0.8) = 0.16

    Arguments:
        df {Pandas.DataFrame}
            Full pandas DataFrame to know the actual features they will use

        xTrain {array} - shape(batchTrain, nFeaturesInput)
            Array of input data

        pValue {float} - threshold value of var() to delete
            Variance threshold value we want to delete

    Output:
        xTrainSelectedFeatures {array}
            shape(batchTrain, nSelectedFeaturesInput)

        xTestSelectedFeatures {array}
            shape(batchTest, nSelectedFeaturesInput)

        selectedFeaturesList {list}
            List with the name of the selected features
    """
    #place the threshold 
    #threshold = pValue*(1-pValue) # this is for binary data/percentage
    threshold = pValue
    selector = VarianceThreshold(threshold=threshold)

    # transform to pandas in order to get the selected features 
    # for training dataset and the first lag day ALWAYS
    xTrain = pd.DataFrame(xTrain[:,0,:], columns=featureList)
    xTrainSelectedFeatures = selector.fit_transform(xTrain)

    # transform to pandas in order to get the selected features 
    # for testing dataset and the first lag day ALWAYS
    xTest = pd.DataFrame(xTest[:,0,:], columns=featureList)
    xTestSelectedFeatures = selector.transform(xTest)

    # get the selected features
    selectedFeaturesList = xTrain.columns[selector.get_support()]

    return np.array(xTrainSelectedFeatures), np.array(xTestSelectedFeatures), selectedFeaturesList

def selectFeatureUnivariate(
    xTrain, 
    xTest, 
    yTrain, 
    pValue, 
    featureList, 
    scoringFunction = "mutual_info_regression", 
    nNeighbors=3):
    """
    Univariate feature selection works by selecting the best features based on univariate 
    statistical tests. In this case, f_regression and mutual_info_regression.

    In this case, those features that does not had a scoring value (from r_regression or 
    mutual_info_regression) higher than p_value, will be deleted

    https://scikit-learn.org/stable/auto_examples/feature_selection/plot_f_test_vs_mi.html#sphx-glr-auto-examples-feature-selection-plot-f-test-vs-mi-py
    """
    
    # transform to pandas in order to get the selected features 
    # for training dataset and the first lag day ALWAYS
    xTrainNew = pd.DataFrame(xTrain[:,0,:], columns=featureList)

    # transform to pandas in order to get the selected features 
    # for testing dataset and the first lag day ALWAYS
    xTestNew = pd.DataFrame(xTest[:,0,:], columns=featureList)

    # determine and filter the relations
    if scoringFunction == "mutual_info_regression":
        score = mutual_info_regression(
            X = xTrainNew, 
            y = yTrain,
            n_neighbors = nNeighbors,
            copy = True)
        selectedFeaturesList = [featureList[i] for i in range(len(featureList)) if score[i]>pValue]
        selector = SelectKBest(mutual_info_regression, k=len(selectedFeaturesList))
    else:
        score = r_regression(
            X = xTrainNew, 
            y = yTrain)
        selectedFeaturesList = [featureList[i] for i in range(len(featureList)) if abs(score[i])>pValue]
        selector = SelectKBest(r_regression, k=len(selectedFeaturesList))

    ic(score)
    xTrainNew = selector.fit_transform(xTrainNew, yTrain)
    xTrainNew = np.array(xTrainNew).reshape(xTrain.shape[0], xTrain.shape[1], len(selectedFeaturesList))

    xTestNew = selector.transform(xTestNew)
    xTestNew = np.array(xTestNew).reshape(xTest.shape[0], xTest.shape[1], len(selectedFeaturesList))

    return xTrainNew, xTestNew, selectedFeaturesList

















