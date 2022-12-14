from warnings import warn
from typing import Optional

from tensorflow.keras import layers, Model, Input

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
            optimizer = optimizer,
        )

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
        
    def _saveArchitectureParametersAsAttributes(self, nHiddenLayers, neuronsPerLayerList, activation, optimizer):

        self.nHiddenLayers = nHiddenLayers
        self.neuronsPerLayerList = neuronsPerLayerList
        self.activation = activation
        self.optimizer = optimizer

    def _checkInputParametersfromBuildModel(self):

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

    def train(self):
        pass

    def optimizationAlgorithm(self):
        pass

    def predict(self):
        pass

    def saveModel(self):
        pass

    def loadModel(self):
        pass

    def savePredictions(self):
        pass



