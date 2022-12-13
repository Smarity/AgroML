from typing import Optional

from tensorflow.keras import layers, Model, Input

from agroml.data import ModelData
from agroml.models import MachineLearningModel

class MultiLayerPerceptron(MachineLearningModel):
    def __init__(self, modelData:ModelData):
        super().__init__(modelData)      

    def __str__(self):
        pass

    def buildModel(
        self,
        hiddenLayers: Optional[int] = 1,
        neuronsPerLayerList: Optional[list] = [10],
        activation: Optional[str] = "relu",
        optimizer: Optional[str] = "Adam",
        ) -> Model:
        """
        
        Parameters
        ----------
        hiddenLayers : Optional[int], optional
            Number of hidden layer in the MLP architecture. The default is 1.
        neuronsPerLayerList : Optional[list], optional
            A list with the number of neurons per layer. The default is [10].
            The dimension of the list must be equal to the number of hidden layers,
            otherwise, a warning is raised
        activation : Optional[str], optional
            Activation function for the hidden layers. The default is "relu". Any of
            the following can be set: 'relu','sigmoid', 'softmax', 'softplus', 'softsign', 'tanh','selu, 
            'elu','exponential' -- https://keras.io/api/layers/activations/
        optimizer : Optional[str], optional
            Optimizer to be used. The default is "Adam". Any of the following can
            be set: 'SGD','RMSprop', 'Adam','Adadelta','Adagrad','Adamax','Nadam','
            Ftrl' -- https://keras.io/api/optimizers/
        """
        inputs = Input(shape=(self.nInputs*self.lagDays, ))
        
        x = layers.Dense(neuronsPerLayerList[0], activation)(inputs)
        for i in range(hiddenLayers-1):
            x = layers.Dense(neuronsPerLayerList[i+1], activation)(x)
        outputs = layers.Dense(self.nOutputs, activation)(x)
         
        self.model = Model(inputs, outputs)
        self.model.compile(optimizer, loss='mse', metrics=['mae'])
        
        return self.model
        

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



