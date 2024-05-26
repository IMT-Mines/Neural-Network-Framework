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
        data.normalizeMinMaxFeatures()
        val splitData = data.split(0.8)

        // Create the model
        val model = NeuralNetwork(
            trainingMethod = StandardTraining(splitData, batchSize = 2),
            loss = CategoricalCrossEntropy,
            optimizer = Adam(learningRate = 0.001, beta1 = 0.9, beta2 = 0.999)
        )
        model.addLayer(Layer(4))
        model.addLayer(Layer(20, LeakyReLU, NormalHeInitialization))
        model.addLayer(Layer(10, LeakyReLU, NormalHeInitialization))
        model.addLayer(Layer(3, Softmax, NormalXavierGlorotInitialization))
        model.initialize()

        // Train and test the model
        model.fit(1000)
        model.save("src/main/resources/irisModel.txt")
        model.test()
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
        data.normalizeMinMaxFeatures()
        val splitData = data.split(0.8)

        // Create the model
        val model = NeuralNetwork(
            trainingMethod = StandardTraining(splitData, batchSize = 64),
            loss = CategoricalCrossEntropy,
            optimizer = Adam(learningRate = 0.001, beta1 = 0.9, beta2 = 0.999)
        )
        model.addLayer(Layer(784))
        model.addLayer(Layer(300, LeakyReLU, NormalHeInitialization))
        model.addLayer(Layer(100, LeakyReLU, NormalHeInitialization))
        model.addLayer(Layer(10, Softmax, NormalXavierGlorotInitialization))
        model.initialize()

        // Train and test the model
        model.fit(50)
        model.save("src/main/resources/digitsModel.txt")
        model.test()
    }

    /**
     * This function is used to classify the wheat seeds dataset
     * The dataset has 7 features and 210 instances
     * The labels are from 1 to 3
     */
    fun seedsClassification() {
        // Load the data
        val data = DataLoader.loadSeeds()
        data.normalizeMinMaxFeatures()
        val splitData = data.split(0.8)

        // Create the model
        val model = NeuralNetwork(
            trainingMethod = StandardTraining(splitData, batchSize = 2),
            loss = CategoricalCrossEntropy,
            optimizer = Adam(learningRate = 0.001, beta1 = 0.9, beta2 = 0.999)
        )
        model.addLayer(Layer(7))
        model.addLayer(Layer(10, ReLU, NormalHeInitialization))
        model.addLayer(Layer(10, ReLU, NormalHeInitialization))
        model.addLayer(Layer(3, Softmax, NormalXavierGlorotInitialization))
        model.initialize()

        // Train and test the model
        model.fit(1000)
        model.save("src/main/resources/seedsModel.txt")
        model.test()
    }
}