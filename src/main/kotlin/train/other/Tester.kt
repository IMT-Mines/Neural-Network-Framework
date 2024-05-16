package main.kotlin.train.other

import main.kotlin.network.*
import main.kotlin.train.Data
import main.kotlin.utils.DebugTools
import main.kotlin.utils.Utils
import kotlin.math.floor


class Tester {

    fun testInitialization() {

        val data = Data()
        data.setDataset(listOf(doubleArrayOf(1.0, 1.0)), listOf())
        Utils.normalizeZScore(data)

        val model = NeuralNetwork(loss = SquaredError, optimizer = SGD(learningRate = 0.01))
        model.addLayer(Layer(2))
        model.addLayer(Layer(2, ReLU, NormalHeInitialization))
        model.addLayer(Layer(1, Sigmoid, NormalXavierGlorotInitialization))
        model.initialize()
        model.save("src/main/resources/model.txt")
//        model.fit(1, )
    }

    fun testBackpropagation() {
        val inputSize = 6
        val outputSize = 3

        val model = NeuralNetwork(loss = CategoricalCrossEntropy, optimizer = SGD(learningRate = 0.001))
        model.addLayer(Layer(inputSize))
        model.addLayer(Layer(1000, ReLU))
        model.addLayer(Layer(1000, ReLU))
        model.addLayer(Layer(outputSize, Softmax))
        model.initialize()

        val debugTools = DebugTools(model)

        val input = DoubleArray(inputSize) { Math.random() }
        val expected = DoubleArray(outputSize) { 0.0 }


        expected[floor(Math.random() * outputSize).toInt()] = 1.0

        val data = Data()
        data.setDataset(listOf(DoubleArray(inputSize) { Math.random() }), listOf(expected))
        model.fit(1000, data)

        model.test(data)


    }
}