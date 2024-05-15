# Neural Network From Scratch

## 1. Features

This "framework" is able to do :

- Multiclass classification
- Binary classification

## 2. Limitations

- Due to the lack of methods, the framework is not able to do regression when the model is too complex

## 3. How to use

In the train package, you can find examples of how to use the framework.

### 3.1 Loading the data

Data should be loaded by the DataLoaders class. If you want to load a new dataset, you should create a method in the
DataLoaders class.
Data can be split and shuffled by the split method (specify the percentage of the training set).

```kotlin
val (train, test) = data.split(0.8)
```

### 3.2 Create the model

The model should be created by the NeuralNetwork class. You must specify the learning rate and the loss function.

```kotlin
val model = NeuralNetwork(0.001, CategoricalCrossEntropy)
```

To add layers, you should use the addLayer method and specify the number of neurons, the activation function and bias.

```kotlin
model.addLayer(10, ReLU, true)
```

**Warning:** The input layer is not implicit (you can just specify the number of neurons).

Finally, you should initialize the model by the initialize method.

### 3.3 Train the model

The model should be trained by the fit method. You must specify the number of epochs and data.

```kotlin
model.fit(100, data)
```

By default, the model uses stochastic gradient descent. Also, it prints the recap of each epoch.
When the training is done, accuracy and loss are printed in ```/resources/plots```.

### 3.4 Use the model

You can try the model with the test method. You must specify the data.

```kotlin
model.test(data)
```

### 3.5 More

You can use DebugTools to print the model's weights and deltas.

```kotlin
var debugTools = DebugTools(model)

// Every x epochs, the weights and deltas are stored
debugTools.archiveWeights()
debugTools.archiveDeltas()

// To print the weights and deltas
debugTools.printWeights()
debugTools.printDeltas()

// You can find plots in /resources/plots/debug
```

## 4. TODO

In the future, I would like to add more features like :

- L1 and L2 regularization
- Adam optimizer
- Initialization methods
- Bias Gestion
- Early stopping
