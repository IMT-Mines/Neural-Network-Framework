package main.kotlin.train.classification

import main.kotlin.network.*
import main.kotlin.train.DataLoader

class MultiClassification {

    /**
     * This function is used to classify the iris dataset
     * If the label is "Iris-setosa" then the label is [1, 0, 0]
     * If the label is "Iris-versicolor" then the label is [0, 1, 0]
     * If the label is "Iris-virginica" then the label is [0, 0, 1]
     * The dataset has 4 features and 150 instances
     */
    fun irisClassification() {
        // Load the data
        val data = DataLoader.loadIris()
        val (train, test) = data.split(0.8)

        // Create the model
        val model = NeuralNetwork(learningRate = 0.001, lossFunction = CategoricalCrossEntropy)
        model.addLayer(Layer(4))
        model.addLayer(Layer(10, ReLU))
        model.addLayer(Layer(10, ReLU))
        model.addLayer(Layer(3, Softmax))
        model.initialize()

        // Train and test the model
        model.fit(1000, train)
        model.save("src/main/resources/irisModel.txt")
        model.test(test)
    }
}