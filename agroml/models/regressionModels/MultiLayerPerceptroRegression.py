import os
from warnings import warn
from typing import Optional

import pandas as pd
from matplotlib import pyplot as plt
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.models import load_model

from agroml.data import ModelData
from agroml.models import MachineLearningModel

class MultiLayerPerceptron(MachineLearningModel):
    def __init__(self, modelData:ModelData):
        super().__init__(modelData)      

    def __str__(self):
        pass

    def __repr__(self):
        printMessage ="""
        MultiLayerPerceptron(
            "hiddenLayers":{}, 
            "neuronsPerLayerList":{}, 
            "activation":{}, 
            "optimizer":{})
        """.format(self.nHiddenLayers, self.neuronsPerLayerList, self.activation, self.optimizer)
        return printMessage

    def __eq__(self, other):
        if self.nHiddenLayers == other.nHiddenLayers and \
           self.neuronsPerLayerList == other.neuronsPerLayerList and \
           self.activation == other.activation and \
           self.optimizer == other.optimizer:
            return True
        else:
            return False

    def buildModel(
        self,
        nHiddenLayers: Optional[int] = 1,
        neuronsPerLayerList: Optional[list] = [10],
        activation: Optional[str] = "relu",
        optimizer: Optional[str] = "Adam",
        ) -> Model:
        """
        
        Parameters
        ----------
        nHiddenLayers : Optional[int], optional
            Number of hidden layer in the MLP architecture. The default is 1.
        neuronsPerLayerList : Optional[list], optional
            A list with the number of neurons per layer. The default is [10].
            The dimension of the list must be equal to the number of hidden layers,
            otherwise, an error will be raised.
        activation : Optional[str], optional
            Activation function for the hidden layers. The default is "relu". Any of
            the following can be set: 'relu','sigmoid', 'softmax', 'softplus', 'softsign', 'tanh','selu, 
            'elu','exponential' -- https://keras.io/api/layers/activations/
        optimizer : Optional[str], optional
            Optimizer to be used. The default is "Adam". Any of the following can
            be set: 'SGD','RMSprop', 'Adam','Adadelta','Adagrad','Adamax','Nadam','
            Ftrl' -- https://keras.io/api/optimizers/
        """
        
        self._saveArchitectureParametersAsAttributes(
            nHiddenLayers = nHiddenLayers,
            neuronsPerLayerList = neuronsPerLayerList,
            activation = activation,
            optimizer = optimizer)

        self._checkInputParametersfromBuildModel()

        inputs = Input(shape=(self.nFeatures, ))
        x = layers.Dense(self.neuronsPerLayerList[0], self.activation)(inputs)
        for i in range(self.nHiddenLayers-1):
            x = layers.Dense(neuronsPerLayerList[i+1], self.activation)(x)
        outputs = layers.Dense(self.nOutputs, self.activation)(x)
         
        self.model = Model(inputs, outputs)
        self.model.compile(self.optimizer, loss='mse', metrics=['mae'])

        self.allBuiltModelsList.append(self.model)
        return self.model
        
    def _saveArchitectureParametersAsAttributes(
        self, 
        nHiddenLayers: int, 
        neuronsPerLayerList: list, 
        activation: str, 
        optimizer:str) -> None:

        self.nHiddenLayers = nHiddenLayers
        self.neuronsPerLayerList = neuronsPerLayerList
        self.activation = activation
        self.optimizer = optimizer

    def _checkInputParametersfromBuildModel(self) -> None:

        if len(self.neuronsPerLayerList) < self.nHiddenLayers:
            raise ValueError("The number of neurons per layer must be equal to the number of hidden layers")
        elif len(self.neuronsPerLayerList) > self.nHiddenLayers:
            warn(UserWarning("The number of neurons per layer is greater than the number of hidden layers. The extra neurons will be ignored"))

        if self.activation not in ["relu","sigmoid", "softmax", "softplus", "softsign", "tanh", "selu", "elu","exponential"]:
            self.activation = "relu"
            warn(UserWarning("The activation function is not valid. The default activation function will be used (relu)"))

        if self.optimizer not in ["SGD","RMSprop", "Adam","Adadelta","Adagrad","Adamax","Nadam","Ftrl"]:
            self.optimizer = "Adam"
            warn(UserWarning("The optimizer is not valid. The default optimizer will be used (Adam)"))

    def train(self, epochs:int, verbose:int=1):
        """
        Parameters:
        ----------
        epochs: int
            Number of epochs to train the model
        verbose: int, optional
            Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.
        """
        self.trainHistory=self.model.fit(
            self.xTrain, 
            self.yTrain, 
            epochs=epochs,
            verbose=verbose)
            #callbacks = self.my_callback)

    def plotTrainHistory(self):
        """ It plots the train history of the model

        Returns:
        ----------
        fig: matplotlib.figure.Figure
            Figure with the train history
        """
        fig = plt.figure()
        plt.plot(self.trainHistory.history['mae'])
        plt.plot(self.trainHistory.history['loss'])
        plt.legend(['MAE','MSE'])
        plt.xlabel('Epochs')
        plt.ylabel('Error')
        plt.title('Train History')

        return fig

    def optimizationAlgorithm(self):
        pass

    def predict(self, xTest:Optional[pd.DataFrame]= None) -> pd.DataFrame:
        """ It predicts the output of the model using the test data

        Returns:
        ----------
        yPred: array
            Array with the predicted values
        """
        if xTest is None:
            yPred = self.model.predict(self.xTest)
        else:

            yPred = self.model.predict(xTest)
        self.yPred = pd.DataFrame(yPred, columns=self.outputsList)
        
    # In future may be using pickle
    def saveModel(self, fileName:str) -> None:
        """
        It saves the model in an specific location and name

        Parameters:
        ----------
        fileName: str
            File name to save model (without extension). The .h5 extension will
            be added automatically
        """
        if fileName.split('.')[-1] != 'h5':
            fileName += '.h5'
        self.model.save(fileName)

    def loadModel(self, fileName:str) -> None:
        """ It loads a specific model 

        Parameters:
        ----------
        fileName: str
            File name to load model (with extension). The .h5 must be included
        """
        self.model = load_model(fileName)

        loadedArchitecture = self._getArchitectureFromLoadedModels()

        self._saveArchitectureParametersAsAttributes(
            nHiddenLayers = loadedArchitecture["nHiddenLayers"], 
            neuronsPerLayerList = loadedArchitecture["neuronsPerLayerList"], 
            activation = loadedArchitecture["activation"], 
            optimizer = loadedArchitecture["optimizer"])
         

    def _getArchitectureFromLoadedModels(self) -> dict:
        architecture = dict()
        
        architecture["nHiddenLayers"] = len(self.model.layers) - 2
        architecture["neuronsPerLayerList"] = [
            self.model.layers[i+1].units for i in range(architecture["nHiddenLayers"])]
        architecture["activation"] = self.model.layers[1].activation.__name__
        architecture["optimizer"] = self.model.optimizer.__class__.__name__
        
        return architecture

    def savePredictions(self, fileName:str, sep:Optional[str]) -> None:
        """It saves the predictions values of a model
        
        Parameters:
        ----------
        fileName: str
            File name to save model (with extension)
        """
        dfPredictions = pd.DataFrame()

        for output in range(self.nOutputs):
            dfPredictions["pred_{}".format(output)] = self.yPred[output]
            dfPredictions["meas_{}".format(output)] = self.yTest[output]

        if self._getFileExtension() in set([".xls", ".xlsx"]):
           dfPredictions.to_excel(fileName)
        else: 
            dfPredictions.to_csv(str(fileName))  

    def _getFileExtension(self) -> str:
        return os.path.splitext(self.fileLocation)[-1]





