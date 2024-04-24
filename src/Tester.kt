class Tester {

    companion object {
        fun testBackpropagation() {
            val model = NeuralNetwork(learningRate = 0.1)
            model.addLayer(Layer(4, Sigmoid, false))
            model.addLayer(Layer(10, Sigmoid, false))
            model.addLayer(Layer(9, Sigmoid, false))
            model.initialize()
            model.save("model.txt")


            val input: DoubleArray = DoubleArray(4) { 0.0 }
            val expected: DoubleArray = DoubleArray(9) { 0.0 }

            for (i in 0..<4) {
                input[i] = Math.random()
            }

            expected[2] = 1.0

            // BEFORE BACKPROPAGATION
            var output = model.predict(input)
            var error = model.meanSquaredError(output, expected)
            println("Mean squared error: $error")
            for (i in expected.indices) {
                println("Output: ${output[i]} : Expected: ${expected[i]} -> Error: ${expected[i] - output[i]}")
            }

            model.optimize(expected, input)

            // AFTER BACKPROPAGATION
            output = model.predict(input)
            error = model.meanSquaredError(output, expected)
            println("Mean squared error: $error")
            for (i in expected.indices) {
                println("Output: ${output[i]} : Expected: ${expected[i]} -> Error: ${expected[i] - output[i]}")
            }

            model.save("model.txt")
        }



        fun testEnvironment() {
        }
    }
}