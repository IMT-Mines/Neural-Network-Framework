package main.kotlin.train.other

import main.kotlin.debug.DebugTools
import main.kotlin.network.*


class Tester {

    fun testBackpropagation() {
        val model = NeuralNetwork(learningRate = 1.0, lossFunction = MeanSquaredError)
        model.addLayer(Layer(6))
        model.addLayer(Layer(10, ReLU))
        model.addLayer(Layer(3, Softmax))
        model.initialize()

        val debugTools = DebugTools(model)

        val input: DoubleArray = doubleArrayOf(0.2, 0.8, 0.5, 0.1, 0.3, 0.1)
        val expected: DoubleArray = doubleArrayOf(0.1, 0.0, 0.0)

        // BEFORE BACKPROPAGATION
        var output = model.predict(input)
        var error = model.lossFunction.loss(output, expected)
        println("Mean squared error: $error")
        for (i in expected.indices) {
            println("Output: ${output[i]} : Expected: ${expected[i]} -> Error: ${expected[i] - output[i]}")
        }

        for (i in 0..<10) {
            debugTools.archiveWeights()
            debugTools.archiveDelta()
            model.predict(input)
            model.stochasticGradientDescent(expected)
        }

        // AFTER BACKPROPAGATION
        output = model.predict(input)
        error = model.lossFunction.loss(output, expected)
        println("Mean squared error: $error")
        for (i in expected.indices) {
            println("Output: ${output[i]} : Expected: ${expected[i]} -> Error: ${expected[i] - output[i]}")
        }
        debugTools.printDeltas()
        debugTools.printWeights()
    }
}