package main.kotlin.train.classification

import main.kotlin.network.*
import main.kotlin.train.DataLoader
import main.kotlin.utils.Utils

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
        Utils.normalizeZScore(data)
        val (train, test) = data.split(0.8)

        // Create the model
        val model = NeuralNetwork(learningRate = 0.01, loss = CategoricalCrossEntropy)
        model.addLayer(Layer(4))
        model.addLayer(Layer(20, LeakyReLU))
        model.addLayer(Layer(10, LeakyReLU))
        model.addLayer(Layer(3, Softmax))
        model.initialize()
        println(model)

        // Train and test the model
        model.fit(1000, train, batchSize = 2)
        model.save("src/main/resources/irisModel.txt")
        model.test(test)
    }

    /**
     * Due to the lack of methods (normalization, convolution, etc.) the model is not able to classify the digits dataset
     * This function is used to classify the digits dataset
     * The dataset has 784 features and 42000 instances
     * The labels are from 0 to 9
     */
    fun digitsClassification() {
        // Load the data
        val data = DataLoader.loadDigits()
        Utils.normalizeZScore(data)
        val (train, test) = data.split(0.8)
        println(train.size())

        // Create the model
        val model = NeuralNetwork(learningRate = 0.001, loss = BinaryCrossEntropy)
        model.addLayer(Layer(784))
        model.addLayer(Layer(784, ReLU))
        model.addLayer(Layer(100, ReLU))
        model.addLayer(Layer(10, Softmax))
        model.initialize()

        // Train and test the model
        model.fit(1, train, batchSize = 64)
        model.save("src/main/resources/digitsModel.txt")
        model.test(test)
    }

    /**
     * This function is used to classify the wheat seeds dataset
     * The dataset has 7 features and 210 instances
     * The labels are from 1 to 3
     */
    fun seedsClassification() {
        // Load the data
        val data = DataLoader.loadSeeds()
        Utils.normalizeZScore(data)
        val (train, test) = data.split(0.8)

        // Create the model
        val model = NeuralNetwork(learningRate = 0.001, loss = CategoricalCrossEntropy)
        model.addLayer(Layer(7))
        model.addLayer(Layer(10, ReLU))
        model.addLayer(Layer(10, ReLU))
        model.addLayer(Layer(3, Softmax))
        model.initialize()

        // Train and test the model
        model.fit(1000, train)
        model.save("src/main/resources/seedsModel.txt")
        model.test(test)
    }
}