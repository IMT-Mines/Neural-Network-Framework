package main.kotlin.train.classification

import main.kotlin.network.*
import main.kotlin.train.DataLoader

class BinaryClassification {

    /**
     * This function is used to classify the sonar dataset
     * If the label is "R" then the output is 1.0
     * If the label is "M" then the output is 0.0
     * The dataset has 60 features and 208 instances
     */
    fun sonarClassification() {
        // Load the data
        val data = DataLoader.loadSonar()
        val (train, test) = data.split(0.8)

        // Create the model
        val model = NeuralNetwork(learningRate = 0.001, loss = BinaryCrossEntropy)
        model.addLayer(Layer(60))
        model.addLayer(Layer(60, LeakyReLU, NormalHeInitialization))
        model.addLayer(Layer(10, LeakyReLU, NormalHeInitialization))
        model.addLayer(Layer(1, Sigmoid, NormalXavierGlorotInitialization))
        model.initialize()

        // Train and test the model
        model.fit(1000, train, batchSize = 1, false)
        model.save("src/main/resources/sonarModel.txt")
        model.test(test)
    }

    /**
     * This function is used to classify the ionosphere dataset
     * If the label is "g" then the output is 1.0
     * If the label is "b" then the output is 0.0
     * The dataset has 33 features and 351 instances
     */
    fun ionosphereClassification() {
        // Load the data
        val data = DataLoader.loadIonosphere()
        data.normalizeMinMaxFeatures()
        val (train, test) = data.split(0.8)

        // Create the model
        val model = NeuralNetwork(learningRate = 0.001, loss = BinaryCrossEntropy)
        model.addLayer(Layer(33))
        model.addLayer(Layer(33, LeakyReLU, NormalHeInitialization))
        model.addLayer(Layer(10, LeakyReLU, NormalHeInitialization))
        model.addLayer(Layer(4, LeakyReLU, NormalHeInitialization))
        model.addLayer(Layer(1, Sigmoid, NormalXavierGlorotInitialization))
        model.initialize()

        // Train and test the model
        model.fit(1000, train, batchSize = 2, true)
        model.save("src/main/resources/ionosphereModel.txt")
        model.test(test)
    }
}