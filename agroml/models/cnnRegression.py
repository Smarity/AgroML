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

class ConvolutionalNeuralNetwork:
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
        
        
        # reshape x data - tensorflow 2.4 does no accept 3D or higher dimensional
        #self.xTrain = np.reshape(self.xTrain,(self.batchTrain,self.lagDays*self.nInputs))
        #self.xTest = np.reshape(self.xTest,(self.batchTest,self.lagDays*self.nInputs))

    def _addPoolLayer(self, model, type_pool='Max', pool_size=2):
        if type_pool == 'Avg':
            model = layers.AvgPool1D(pool_size = pool_size)(model)
        elif type_pool == 'Max':
            model = layers.MaxPool1D(pool_size = pool_size)(model)
        return model
        
    def buildModel(
        self, 
        nBlock, 
        nConvolutions, 
        nFilters, 
        nKernels, 
        typePool, 
        poolSize, 
        hiddenLayers, 
        neurons, 
        activation, 
        optimizer):
        '''
        Arguments:
            nBlock {int} - Number of blocks (conv + pool)
            nConvBase {int} - Number of convolutions in the block
            nFilters {int} - Number of output filters in convolution layer
            nKernels {int} - Kernel length
            typePool {str} - 'Avg' or 'Max'
            poolSize {int} - Size pooling window
            hiddenLayers {int} - Number of hidden layers in final MLP
            neurons {int} - Number of neurons in each hidden layer
            activation {str} - Activation function ['relu','sigmoid', 'softmax',
                'softplus', 'softsign', 'tanh','selu, 'elu','exponential']
                https://keras.io/api/layers/activations/
            optimizer {str} - Optimizer function to learning model ['SGD','RMSprop',
                'Adam','Adadelta','Adagrad','Adamax','Nadam','Ftrl']
                https://keras.io/api/optimizers/
        
        '''
        inputs = Input(shape=(self.lagDays, self.nInputs))

        x = layers.Conv1D(filters=nFilters, kernel_size=(nKernels,), padding="same", activation=activation)(inputs)
        for i in range(nBlock-1):
            for j in range(nConvolutions):
                x = layers.Conv1D(filters=nFilters, kernel_size=(nKernels,), padding="same", activation=activation)(x)
            x = self._addPoolLayer(x, type_pool=typePool, pool_size=int(poolSize))
        x = layers.Flatten()(x)

        for i in range(hiddenLayers):
            x = layers.Dense(units=neurons, activation = activation)(x)
        outputs = layers.Dense(self.nOutputs, activation=activation)(x)

        model = Model(inputs, outputs)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        return model

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
            yPred = self.predictFullTrainingData(model)
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
        nBlockList = [1, 3], 
        nConvolutionsList = [1, 3], 
        nFiltersList = [8, 64], 
        nKernelsList = [8, 64], 
        typePoolList = ['Avg', 'Max'], 
        poolSizeList= [2, 15],
        hiddenLayersList=[1,5], 
        neuronsList=[5, 25], 
        activationList=['relu'], 
        optimizerList=['adam'], 
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
            hiddenLayersList (list) - min. and max. number of hidden layers 
            neuronsList (list) - min. and max. number of neuron in the hidden layers
            activationList (list) - list of activation functions ['relu','sigmoid', 
                'softmax','softplus', 'softsign', 'tanh','selu', 'elu','exponential']
            optimizerList (list) - list of optimizers ['SGD','RMSprop','Adam',
                'Adadelta','Adagrad','Adamax','Nadam','Ftrl']
            epochsList (list) - min. and max. number of epochs during training, 
            bayesianEpochs (int) - number of total epochs in Bayesian Optimization
            randomStart (int) - number of random epochs in Bayesian Optimization
            validationSplit - float (from 0 to 1) with the percentaje of validation dataset
            shuffle=False (bool) - If True, it shuffles the data of validation split
                (only on holdout, not crossVal)
            crossVal (bool) - If True, it carries out a cross validation
            nFolds (int) - Number of folds in cross validation
        '''

        # creation of the hyperparameter space
        # creation of the dimensions of the variables to study
        dim_nBlock = Integer(low=nBlockList[0], high=nBlockList[1], name='nBlock')
        dim_nConvolutions = Integer(low=nConvolutionsList[0], high=nConvolutionsList[1], name='nConvolutions')
        dim_nFilters = Integer(low=nFiltersList[0], high=nFiltersList[1], name='nFilters')
        dim_nKernels = Integer(low=nKernelsList[0], high=nKernelsList[1], name='nKernels')
        dim_typePool = Categorical(categories=typePoolList, name = 'typePool')
        dim_poolSize = Integer(low=poolSizeList[0], high=poolSizeList[1], name='poolSize')
        dim_hiddenLayers = Integer(low=hiddenLayersList[0], high=hiddenLayersList[1], name='hiddenLayers')
        dim_neurons = Integer(low=neuronsList[0], high=neuronsList[1], name='neurons')
        dim_activation = Categorical(categories=activationList, name='activation')
        dim_optimizer = Categorical(categories=optimizerList, name = 'optimizer')
        dim_epochs = Integer(low=epochsList[0], high=epochsList[1], name='epochs')
        
        self.dimensions = [
            dim_nBlock, dim_nConvolutions, dim_nFilters, dim_nKernels, dim_typePool, dim_poolSize,
            dim_hiddenLayers, dim_neurons, dim_activation, dim_optimizer, dim_epochs
        ]
        
        # default parameters
        self.defaultParameters = [
            nBlockList[0], nConvolutionsList[0], nFiltersList[0], nKernelsList[0], typePoolList[0], 
            poolSizeList[0], hiddenLayersList[0], neuronsList[0], activationList[0], optimizerList[0], epochsList[0]
        ]
        
        # fitness function
        @use_named_args(dimensions= self.dimensions)
        def fitnessFunction(**params):
            
            try:
                # build model
                model = self.buildModel(
                    nBlock = params['nBlock'], 
                    nConvolutions = params['nConvolutions'], 
                    nFilters = params['nFilters'], 
                    nKernels = params['nKernels'], 
                    typePool = params['typePool'], 
                    poolSize = params['poolSize'], 
                    hiddenLayers = params['hiddenLayers'], 
                    neurons=params['neurons'],
                    activation=params['activation'],
                    optimizer=params['optimizer'])
            
                mae = self._fitAndValidationAssess(
                    model, validationSplit, shuffle, crossVal, nFolds, params['epochs'])
                
            except:
                mae=1000
            
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
            nBlock = bayesianBestParameters.x[0], 
            nConvolutions = bayesianBestParameters.x[1], 
            nFilters = bayesianBestParameters.x[2], 
            nKernels = bayesianBestParameters.x[3], 
            typePool = bayesianBestParameters.x[4], 
            poolSize = bayesianBestParameters.x[5], 
            hiddenLayers = bayesianBestParameters.x[6], 
            neurons = bayesianBestParameters.x[7],
            activation = bayesianBestParameters.x[8],
            optimizer = bayesianBestParameters.x[9])
        
        # best parameters
        bestParams = {
            'nBlock': bayesianBestParameters.x[0], 
            'nConvolutions': bayesianBestParameters.x[1], 
            'nFilters': bayesianBestParameters.x[2], 
            'nKernels': bayesianBestParameters.x[3], 
            'typePool': bayesianBestParameters.x[4], 
            'poolSize': bayesianBestParameters.x[5], 
            'hiddenLayers': bayesianBestParameters.x[6], 
            'neurons': bayesianBestParameters.x[7],
            'activation': bayesianBestParameters.x[8],
            'optimizer': bayesianBestParameters.x[9],
            'epochs': bayesianBestParameters.x[10]
        }

        return model, bestParams
        
        
        
        
        
