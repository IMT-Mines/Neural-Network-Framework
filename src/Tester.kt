class Tester {

    companion object {
        fun testBackpropagation() {
            val model = NeuralNetwork(learningRate = 0.01)
            model.addLayer(Layer(5, Sigmoid, false))
            model.addLayer(Layer(10, Sigmoid, false))
            model.addLayer(Layer(5, Sigmoid, false))
            model.initialize()
            model.save("model.txt")


            val input = doubleArrayOf(1.5, 2.0, 3.0, 4.0, 5.0)
            val expected = doubleArrayOf(0.5, 0.2, 1.0, 0.4, 0.13)

            // BEFORE BACKPROPAGATION
            var output = model.predict(input)
            var error = model.meanSquaredError(output, expected)
            println("Mean squared error: $error")
            for (i in expected.indices) {
                println("Input: ${input[i]} -> Output: ${output[i]} : Expected: ${expected[i]} -> Error: ${expected[i] - output[i]}")
            }

            model.backpropagation(expected, input)

            // AFTER BACKPROPAGATION
            output = model.predict(input)
            error = model.meanSquaredError(output, expected)
            println("Mean squared error: $error")
            for (i in expected.indices) {
                println("Input: ${input[i]} -> Output: ${output[i]} : Expected: ${expected[i]} -> Error: ${expected[i] - output[i]}")
            }
        }
    }
}