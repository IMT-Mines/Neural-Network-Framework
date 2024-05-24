package main.kotlin.train.other

import main.kotlin.network.*
import main.kotlin.train.Data
import kotlin.math.floor


class Tester {

    fun testInitialization() {
        val data = Data()
        data.setDataset(listOf(doubleArrayOf(1.0, 1.0)), listOf())
        data.normalizeZFeatures()

        val model = NeuralNetwork(
            trainingMethod = StandardTraining(Pair(data, data)),
            loss = SquaredError,
            optimizer = SGD(learningRate = 0.01)
        )
        model.addLayer(Layer(2))
        model.addLayer(Layer(2, ReLU, NormalHeInitialization))
        model.addLayer(Layer(1, Sigmoid, NormalXavierGlorotInitialization))
        model.initialize()
        model.save("src/main/resources/model.txt")
    }

    fun testBackpropagation() {
        val inputSize = 6
        val outputSize = 3

        val input = DoubleArray(inputSize) { Math.random() }
        val expected = DoubleArray(outputSize) { 0.0 }
        expected[floor(Math.random() * outputSize).toInt()] = 1.0

        val data = Data()
        data.setDataset(listOf(DoubleArray(inputSize) { Math.random() }), listOf(expected))

        val model = NeuralNetwork(
            trainingMethod = StandardTraining(Pair(data, data)),
            loss = CategoricalCrossEntropy,
            optimizer = SGD(learningRate = 0.001)
        )
        model.addLayer(Layer(inputSize))
        model.addLayer(Layer(1000, ReLU))
        model.addLayer(Layer(1000, ReLU))
        model.addLayer(Layer(outputSize, Softmax))
        model.initialize()

        model.fit(1000)
        model.test()
    }
}