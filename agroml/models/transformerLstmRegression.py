import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from skopt.utils import use_named_args
from skopt.space import Real, Categorical, Integer
from skopt import gp_minimize
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from icecream import ic

from agroml.utils.statistics import *

class transformerLSTM:
    def __init__(
        self, 
        xTrain, 
        xTest, 
        yTrain, 
        yTest):
        """
        Arguments:
            xTrain {array or np.array} - shape(bath, lagDays, nFeaturesInput)
            
            xTest {array} - shape(bath, lagDays, nFeaturesInput)
            
            yTrain {array} - shape(bath, nFeaturesOutput)
            
            yTest {array} - shape(bath, nFeaturesOutput)
        """
        # input data as float
        try:
            self.xTrain = np.array(xTrain).astype(float)
            self.xTest = np.array(xTest).astype(float)
            self.yTrain = np.array(yTrain).astype(float)
            self.yTest = np.array(yTest).astype(float)
        except:
            print("There exist non-numeric values")
            raise
        
        # shape is not correct
        assert len(self.xTrain.shape)==3, 'xTrain shape is wrong'
        assert len(xTest.shape)==3, 'xTrain shape is wrong'
        assert len(yTrain.shape)==2, 'yTrain shape is wrong'
        assert len(yTest.shape)==2, 'yTrain shape is wrong'
        
        # all lagDays must be the same
        assert xTrain.shape[1]==xTest.shape[1]
        
        self.nInputs = self.xTrain.shape[2]
        self.nOutputs = self.yTrain.shape[1]
        self.lagDays = self.xTrain.shape[1]
        self.batchTrain = self.xTrain.shape[0]
        self.batchTest = self.xTest.shape[0]

    def _transformerEncoder(
        self, 
        inputs, 
        headSize, 
        numHeads, 
        lstmUnits, 
        dropout=0,
        lstmActivation='tanh',
        recurrentActivation='sigmoid'):
        
        # Normalization and Attention
        x = layers.MultiHeadAttention(key_dim=headSize, num_heads=numHeads, dropout=dropout)(inputs, inputs)
        x = layers.Dropout(dropout)(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        res = x + inputs

        # Feed Forward Part
        x = layers.LSTM(units=lstmUnits, activation = lstmActivation, recurrent_activation = recurrentActivation, use_bias=True, unroll=False, recurrent_dropout = 0,  return_sequences=True)(res)
        x = layers.Dropout(dropout)(x)
        x = layers.LSTM(units=inputs.shape[-1], activation = lstmActivation, recurrent_activation = recurrentActivation, use_bias=True, unroll=False, recurrent_dropout = 0,  return_sequences=True)(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x)

        return x + res
    
    def buildModel(
        self, 
        headSize, 
        nHeads, 
        lstmUnits, 
        nBlocks, 
        nHiddenLayers, 
        nHiddenNeurons, 
        attentionDropout=0, 
        mlpDropout=0,
        mlpActivation='relu'):
        
        inputs = Input(shape=(self.lagDays, self.nInputs))
        # loop of transformers
        x = self._transformerEncoder(inputs, headSize, nHeads, lstmUnits, dropout=attentionDropout)
        for _ in range(nBlocks-1):
            x = self._transformerEncoder(x, headSize, nHeads, lstmUnits, dropout=attentionDropout)

        # final MLP to regression
        x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
        for _ in range(nHiddenLayers):
            x = layers.Dense(nHiddenNeurons, activation=mlpActivation)(x)
            x = layers.Dropout(mlpDropout)(x)

        outputs = layers.Dense(self.nOutputs, activation=mlpActivation)(x)
        self.model = Model(inputs, outputs)
        self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])

        return self.model

    def saveModel(self, model, fileName):
        """
        It saves the model in an specific location and name

        Arguments:
            model {keras.model} - Model to save 
            fileName {str} - file name to save model (without extension)
        """
        model.save(fileName+'.h5')

    def loadModel(self, fileName):
        """
        It loads a specific model 

        Arguments:
            fileName {str} - file name to save model (with extension)

        Output:
            model {keras model} - It returns the model
        """
        model = load_model(fileName)
        return model
        
    def savePredictions(self, yPred, yTest):
        """
        It saves the predictions values of a model

        Arguments:
            yPred {array} - Array with the predictions. The shape must be
                (batch, number of output features)

            yTest {array} - Array with the measured values. The shape must be
                (batch, number of output features)

            fileName {str} - file name to save model (with extension)

        Output:
            None
        """
        dfPredictions = pd.DataFrame()

        for i in range(self.nOutputs):
            dfPredictions["pred_{}".format(i)] = yPred[:, i]
            dfPredictions["meas_{}".format(i)] = yTest[:, i]

        dfPredictions.to_csv(str(nameFile)+".csv")

    
    def trainFullTrainingData(
        self, 
        model, 
        epochs=100, 
        verbose=0, 
        showGraph=False):
        '''
        It trains the model, previously build by 'buildModel' or 
        'bayesianOptimization'

        Arguments:
            verbose {int} - 1: Show details / 0: Do not show details
            showGraph {Bool} - True: See training graph / False: Do not see training graph
        '''
        
        history=model.fit(
            self.xTrain, 
            self.yTrain, 
            epochs=epochs,
            verbose=verbose)
            #callbacks = self.my_callback)

        if showGraph:
            #plt.figure(figsize=(10,5), dpi=500)
            plt.figure()
            plt.plot(history.history['mae'])
            plt.plot(history.history['loss'])
            plt.legend(['MAE','MSE'])
            plt.show()
            
        return model

    def predictFullTestingData(
        self, 
        model):
        '''
        It returns a numpy array with the predictions with the shape:
        '''
        pred = np.array(model.predict(self.xTest))
        pred = pred.reshape(self.batchTest, self.nOutputs)
        return pred 
                
    def trainModel(
        self, 
        model, 
        xTrain, 
        yTrain, 
        epochs, 
        verbose=0, 
        showGraph=False):
        '''
        It trains the model, previously build by 'build_model' or 
        'bayesian_optimization' using specific data

        Arguments:
            verbose {int} - 1: Show details / 0: Do not show details
            showGraph {Bool} - True: See training graph / False: Do not see training graph
        '''
        
        history=model.fit(
            xTrain, 
            yTrain, 
            epochs=epochs,
            verbose=verbose)
            #callbacks = self.my_callback)

        if showGraph:
            #plt.figure(figsize=(10,5), dpi=500)
            plt.figure()
            plt.plot(history.history['mae'])
            plt.plot(history.history['loss'])
            plt.legend(['MAE','MSE'])
            plt.show()
            
        return model
    
    def predictModel(
        self, 
        model, 
        xTest):
        '''
        It returns a numpy array with the predictions with the shape:
        
        Arguments:
            model
            xTest (np.array) - InputData to predict
        '''
        pred = np.array(model.predict(xTest))
        pred = pred.reshape(xTest.shape[0], self.nOutputs)
        return pred   
    
    def _fitAndValidationAssess(
        self, 
        model, 
        validationSplit, 
        shuffle, 
        crossVal, 
        nFolds, 
        epochs):
        """
        It splits the training data into validation using different techniches, holdout and crossvalidation.
        In the holdout
        """
        
        if crossVal:
            kf = KFold(n_splits=nFolds)
            maeList = []
            for train_index, val_index in kf.split(self.xTrain):
                bayes_xTrain = self.xTrain[train_index]
                bayes_xVal = self.xTrain[val_index]
                bayes_yTrain = self.yTrain[train_index]
                bayes_yVal = self.yTrain[val_index]
                
                model = self.trainModel(model, bayes_xTrain, bayes_yTrain, epochs)
                bayes_yPred = self.predictModel(model, bayes_xVal)
                
                mae = getMeanAbsoluteError(bayes_yVal, bayes_yPred)
                mae = np.min(mae)
                maeList.append(mae)
                # clear keras session
                K.clear_session()

            mae_np = np.array(maeList)
            mae = np.mean(mae_np)
            return mae

        elif validationSplit == 0.0:
            model = self.trainFullTrainingData(model, epochs)
            yPred = self.predictFullTestingData(model)
            mae = getMeanAbsoluteError(self.yTest, yPred)
            mae = np.min(mae)
            # clear keras session
            K.clear_session()
            return mae

        else:
            if shuffle==True:
                bayes_xTrain, bayes_xVal, bayes_yTrain, bayes_yVal = train_test_split(
                    self.xTrain,
                    self.yTrain,
                    random_state = 42,
                    test_size = validationSplit)
            else:
                # validation index
                validationIndex = int((1-validationSplit)*len(self.xTrain))
                # get training data to validation
                if self.nInputs == 1:
                    bayes_xTrain = self.xTrain[:validationIndex]
                    bayes_xVal = self.xTrain[validationIndex:]
                else:
                    bayes_xTrain = self.xTrain[:validationIndex,:]
                    bayes_xVal = self.xTrain[validationIndex:,:]
                # get validation data
                if self.nOutputs == 1:
                    bayes_yTrain = self.yTrain[:validationIndex]
                    bayes_yVal = self.yTrain[validationIndex:]
                else:
                    bayes_yTrain = self.yTrain[:validationIndex,:]
                    bayes_yVal = self.yTrain[validationIndex:,:]

            model = self.trainModel(model, bayes_xTrain, bayes_yTrain, epochs)
            bayes_yPred = self.predictModel(model, bayes_xVal)
            mae = getMeanAbsoluteError(bayes_yVal, bayes_yPred)
            mae = np.min(mae)
            
            # clear keras session
            K.clear_session()
            return mae
    
    def bayesianOptimization(
        self, 
        headSizeList=[1, 32], 
        nHeadsList=[1, 32], 
        nBlocksList=[1, 3], 
        lstmUnitsList=[16, 32],
        nHiddenLayersList=[1, 3], 
        nHiddenNeuronsList=[1, 10], 
        attentionDropoutList=[0, 0.15], 
        mlpDropoutList=[0, 0.15],
        mlpActivationList=['relu'],
        epochsList=[50,150], 
        bayesianEpochs=50, 
        randomStart=40, 
        validationSplit=0.0, 
        shuffle=False, 
        crossVal= False, 
        nFolds = 4):
        '''
        It tunes the different hyperparameters using bayesian optimization
        
        Arguments:
            headSizeList: (list of ints) -> [minval, maxval]
                Its value determines the size of each attention head for query and key 
            nHeadsList: (list of ints) -> [minval, maxval]
                Its value determines the number of attention heads
            nBlocksList: (list of ints) -> [minval, maxval]
                It represent how many transformer blocks do we have... originally was 6... 
            lstmUnitsList: (list of ints) -> [minval, maxval]
                The dimensionality of the lstm layer.
            nHiddenLayersList: (list of ints) -> [minval, maxval]
                number of hidden layer in the last mlp model
            nHiddenNeuronsList: (list of ints) -> [minval, maxval]
                number of hidden neurons in last mlp model
            attentionDropoutList: (list of floats) -> [minVal, maxVal]
                The dropout layer randomly puts inputs units to 0. It is located inside the transformer architecture
            mlpDropoutList: (list of floats) -> [minVal, maxVal]
                The dropout layer randomly puts inputs units to 0. It is located after the final MLP model
            mlpActivationList: (list of str) ->['activationFunction1', 'activationFunction2',...]
                activation function used along the architecture
            bayesianEpochs (int) - number of total epochs in Bayesian Optimization
            randomStart (int) - number of random epochs in Bayesian Optimization
            validationSplit - float (from 0 to 1) with the percentaje of validation dataset
            shuffle=False (bool) - If True, it shuffles the data of validation split
                (only on holdout, not crossVal)
            crossVal (bool) - If True, it carries out a cross validation
            nFolds (int) - Number of folds in cross validation
        '''
        
        # creation of the hyperparameter space
        dim_headSize = Integer(low=headSizeList[0], high=headSizeList[1], name='headSize')
        dim_nHeads = Integer(low=nHeadsList[0], high=nHeadsList[1], name='nHeads')
        dim_lstmUnits = Integer(low=lstmUnitsList[0], high=lstmUnitsList[1], name='lstmUnits')
        dim_nBlocks = Integer(low=nBlocksList[0], high=nBlocksList[1], name='nBlocks')
        dim_nHiddenLayers = Integer(low=nHiddenLayersList[0], high=nHiddenLayersList[1], name='nHiddenLayers')
        dim_nHiddenNeurons = Integer(low=nHiddenNeuronsList[0], high=nHiddenNeuronsList[1], name='nHiddenNeurons')
        dim_attentionDropout = Real(low=attentionDropoutList[0], high=attentionDropoutList[1], name='attentionDropout')
        dim_mlpDropout = Real(low=mlpDropoutList[0], high=mlpDropoutList[1], name='mlpDropout')
        dim_mlpAactivation = Categorical(categories=mlpActivationList, name='mlpActivation')
        dim_epochs = Integer(low=epochsList[0], high=epochsList[1], name='epochs')

        
        self.dimensions = [
            dim_headSize, dim_nHeads, dim_lstmUnits, dim_nBlocks, dim_nHiddenLayers, dim_nHiddenNeurons,
            dim_attentionDropout, dim_mlpDropout, dim_mlpAactivation ,dim_epochs]
        
        # default parameters
        self.defaultParameters = [
            headSizeList[0], nHeadsList[0], lstmUnitsList[0], nBlocksList[0], nHiddenLayersList[0], 
            nHiddenNeuronsList[0], attentionDropoutList[0], mlpDropoutList[0], mlpActivationList[0],
            epochsList[0]]
        
        # fitness function
        @use_named_args(dimensions= self.dimensions)
        def fitnessFunction(**params):
        
            # build model
            model = self.buildModel(
                headSize = params['headSize'],
                nHeads = params['nHeads'],
                lstmUnits = params['lstmUnits'],
                nBlocks = params['nBlocks'],
                nHiddenLayers = params['nHiddenLayers'],
                nHiddenNeurons = params['nHiddenNeurons'],
                attentionDropout = params['attentionDropout'],
                mlpDropout = params['mlpDropout'],
                mlpActivation = params['mlpActivation'])
        
            mae = self._fitAndValidationAssess(
                model, validationSplit, shuffle, crossVal, nFolds, params['epochs'])
            return mae
        
            # Bayesian optimization
        bayesianBestParameters = gp_minimize(
            func=fitnessFunction,       # the function to minimize
            dimensions=self.dimensions,  # the bounds on each dimension of x
            n_calls=bayesianEpochs,     # the number of evaluations of f
            n_jobs=-1,                   # Number of cores to run in parallel while running
            x0=self.defaultParameters,  #Initial input points
            n_random_starts=randomStart)
        
        self.bayesianIterations = bayesianBestParameters.x_iters
        
        # best model
        model = self.buildModel(
            headSize = bayesianBestParameters.x[0], 
            nHeads = bayesianBestParameters.x[1], 
            nBlocks = bayesianBestParameters.x[2], 
            lstmUnits = bayesianBestParameters.x[3], 
            nHiddenLayers = bayesianBestParameters.x[4], 
            nHiddenNeurons = bayesianBestParameters.x[5], 
            attentionDropout = bayesianBestParameters.x[6], 
            mlpDropout = bayesianBestParameters.x[7],
            mlpActivation = bayesianBestParameters.x[8])
        
        # best parameters
        bestParams = {
            'headSize': bayesianBestParameters.x[0], 
            'nHeads': bayesianBestParameters.x[1], 
            'nBlocks': bayesianBestParameters.x[2], 
            'lstmUnits': bayesianBestParameters.x[2], 
            'nHiddenLayers': bayesianBestParameters.x[4], 
            'nHiddenNeurons': bayesianBestParameters.x[5], 
            'attentionDropout': bayesianBestParameters.x[6], 
            'mlpDropout': bayesianBestParameters.x[7],
            'mlpActivation': bayesianBestParameters.x[8],
            'epochs': bayesianBestParameters.x[9]
        }

        return model, bestParams
    
    