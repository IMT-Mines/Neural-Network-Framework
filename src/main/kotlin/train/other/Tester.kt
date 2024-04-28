package main.kotlin.train.other

import main.kotlin.network.BinaryCrossEntropy
import main.kotlin.network.Layer
import main.kotlin.network.NeuralNetwork
import main.kotlin.network.Sigmoid


class Tester {

    fun testBackpropagation() {
        val model = NeuralNetwork(learningRate = 0.001, lossFunction = BinaryCrossEntropy)
        model.addLayer(Layer(33))
        model.addLayer(Layer(2, Sigmoid, false))
        model.addLayer(Layer(5, Sigmoid, false))
        model.initialize()
        model.save("model.txt")


        val input: DoubleArray = DoubleArray(33) { 0.0 }
        val expected: DoubleArray = DoubleArray(5) { 0.0 }

        for (i in 0..<33) {
            input[i] = Math.random()
        }

        for (i in 0..<5) {
            expected[i] = Math.random()
        }

        // BEFORE BACKPROPAGATION
        var output = model.predict(input)
        var error = model.lossFunction.totalLoss(output, expected)
        println("Mean squared error: $error")
        for (i in expected.indices) {
            println("Output: ${output[i]} : Expected: ${expected[i]} -> Error: ${expected[i] - output[i]}")
        }

        model.compile(expected)

        // AFTER BACKPROPAGATION
        output = model.predict(input)
        error = model.lossFunction.totalLoss(output, expected)
        println("Mean squared error: $error")
        for (i in expected.indices) {
            println("Output: ${output[i]} : Expected: ${expected[i]} -> Error: ${expected[i] - output[i]}")
        }

        model.save("model.txt")
    }
}