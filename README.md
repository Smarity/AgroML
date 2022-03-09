![AgroML logo](https://github.com/Smarity/AgroML/tree/main/agroml/images/smarityLogo.png)

**Agro**nomy **M**achine **L**earning 
An easy tool for developing estimations and forecasts of agro-meteorological 
variables (solar radiation, reference evapotranspiration, precipitation, 
aridity index) using different machine learning models.

---

AgroML is a ML package for both end-users and researches.
It provides optimized machine learning pipelines given specific time series 
dataset and features (all features must be numerical). A machine learning
pipeline contains data preprocessing (normalization and standardization), 
machine learning algorithms (Random Forest, Extreme Learning Machine, Support
Vector Machine, Multilayer Perceptrons, Convolutional Neural Networks,
Long Short-Term Memory and Transformer-based) and hyperparameter tuning algorithms
(Bayesian optimization).

At the moment, AgroML is restricted to regression problems and single modeling,
although future versions will include classification tasks, as well as ensemble 
modeling and automated feature selection. Focusing on the main issues of 
[automated machine learning](https://link.springer.com/book/10.1007/978-3-030-05318-5)

## Installing AgroML

You can install AgroML with pip: `pip install agroml` (in process)

## Minimal Example

The following example uses AgroML to estimate Reference Evapotranspiration in one of the stations from the testing data

```python
from agroml.utils.splitDataByYear import splitDataByYear
from agroml.models.mlpRegression import MultiLayerPerceptron
from agroml.utils.statistics import *
from agroml.utils.plots import *

if __name__ == '__main__':
   

  # import dataset from example
  df = pd.read_csv('tests/test-data/data-example.csv', sep=';')

  # get important variables
  uniqueStations = np.unique(df['station'])
  uniqueYears = np.unique(df['year'])
  varListInputs = ['tx', 'tn', 'rs', 'day']
  varListOutputs = ['et0']

  # split data to train and test
  xTrain, xTest, yTrain, yTest = splitDataByYear(
      df=df,
      station=uniqueStations[-1], 
      yearTestStart=uniqueYears[-3], 
      varListInputs=varListInputs, 
      varListOutputs=varListOutputs,
      preprocessing = 'standardization')
        
  # create model
  mlModel = MultiLayerPerceptron(xTrain, xTest, yTrain, yTest)

  # Hiperparameter optimization using Bayesian optimization
  mlModelBayes, bestParams = mlModel.bayesianOptimization(
      hiddenLayersList=[1,2], 
      neuronsList=[1, 20], 
      activationList=['relu'], 
      optimizerList=['adam'], 
      epochsList=[50,100], 
      bayesianEpochs=5, 
      randomStart=4, 
      validationSplit=0.2, 
      shuffle=False)

  # train best model with the full dataset
  mlModel.trainFullTrainingData(mlModelBayes, showGraph=False)
  yPred = mlModel.predictFullTestingData(mlModelBayes)
  mbe = getMeanBiasError(yTest, yPred)
  rmse = getRootMeanSquaredError(yTest, yPred)
  nse = getNashSuteliffeEfficiency(yTest, yPred)

  # plot predictions vs. measured
  plotGraphLinealRegresion(
      x = yTest, 
      xName = 'Measures values', 
      y = yPred, 
      yName = 'Predicted values')

```

_note_: By default, GamaClassifier optimizes towards `log_loss`.

## Citing

If you want to cite AgroML, please use [our JOSS publication]().

```latex
@article{Bellido-Jimenez2022,
  doi = {https://doi.org/10.3390/agronomy12030656},
  url = {https://www.mdpi.com/2073-4395/12/3/656},
  year  = {2022},
  month = {march},
  publisher = {Agronomy},
  volume = {12},
  number = {3},
  pages = {},
  author = {Juan Antonio Bellido-Jiménez, Javier Estévez, Joaquin Vanschoren and Amanda Penélope García-Marín},
  title = {{AgroML}: An Open-Source Repository to Forecast Reference Evapotranspiration in Different Geo-Climatic Conditions Using Machine Learning and Transformer-Based Models },
  journal = {Agronomy}
}
```
