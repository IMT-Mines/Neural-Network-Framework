package main.kotlin.train.other

import main.kotlin.debug.DebugTools
import main.kotlin.network.*
import main.kotlin.train.Data
import kotlin.math.floor


class Tester {

    fun testBackpropagation() {
        val inputSize = 6
        val outputSize = 3

        val model = NeuralNetwork(learningRate = 0.001, lossFunction = CategoricalCrossEntropy)
        model.addLayer(Layer(inputSize))
        model.addLayer(Layer(1000, ReLU))
        model.addLayer(Layer(1000, ReLU))
        model.addLayer(Layer(outputSize, Softmax))
        model.initialize()

        val debugTools = DebugTools(model)

        val input = DoubleArray(inputSize) { Math.random()}
        val expected = DoubleArray(outputSize) { 0.0 }


        expected[floor(Math.random() * outputSize).toInt()] = 1.0

        val data = Data()
        data.setDataset(listOf(DoubleArray(inputSize) { Math.random()}), listOf(expected))
        model.fit(1000, data)

        model.test(data)


    }
}