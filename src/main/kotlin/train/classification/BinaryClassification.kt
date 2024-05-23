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
        data.normalizeMinMaxFeatures()
        val splitData = data.split(0.8)

        // Create the model
        val model = NeuralNetwork(
            trainingMethod = StandardTraining(splitData, batchSize = 1),
            loss = BinaryCrossEntropy, optimizer = SGD(learningRate = 0.001)
        )
        model.addLayer(Layer(60))
        model.addLayer(Layer(60, LeakyReLU, NormalHeInitialization))
        model.addLayer(Layer(10, LeakyReLU, NormalHeInitialization))
        model.addLayer(Layer(1, Sigmoid, NormalXavierGlorotInitialization))
        model.initialize()

        // Train and test the model
        model.fit(1000)
        model.save("src/main/resources/sonarModel.txt")
        model.test()
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
        val splitData = data.split(0.8)

        // Create the model
        val model = NeuralNetwork(
            trainingMethod = StandardTraining(splitData, batchSize = 1),
            loss = BinaryCrossEntropy,
            optimizer = Adam(learningRate = 0.001, beta1 = 0.9, beta2 = 0.999)
        )
        model.addLayer(Layer(33))
        model.addLayer(Layer(33, LeakyReLU, NormalHeInitialization))
        model.addLayer(Layer(10, LeakyReLU, NormalHeInitialization))
        model.addLayer(Layer(4, LeakyReLU, NormalHeInitialization))
        model.addLayer(Layer(1, Sigmoid, NormalXavierGlorotInitialization))
        model.initialize()

        // Train and test the model
        model.fit(1000)
        model.save("src/main/resources/ionosphereModel.txt")
        model.test()
    }
}