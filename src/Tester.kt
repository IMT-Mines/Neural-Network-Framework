class Tester {

    companion object {
        fun testBackpropagation() {
            val model = NeuralNetwork(learningRate = 0.001, lossFunction = BinaryCrossEntropy)
            model.addLayer(Layer(33))
            model.addLayer(Layer(10, Sigmoid, false))
            model.addLayer(Layer(1, Sigmoid, false))
            model.initialize()
            model.save("model.txt")


            val input: DoubleArray = DoubleArray(33) { 0.0 }
            val expected: DoubleArray = DoubleArray(1) { 0.0 }

            for (i in 0..<33) {
                input[i] = Math.random()
            }

            expected[0] = 1.0

            // BEFORE BACKPROPAGATION
            var output = model.predict(input)
            var error = model.lossFunction.totalLoss(output, expected)
            println("Mean squared error: $error")
            for (i in expected.indices) {
                println("Output: ${output[i]} : Expected: ${expected[i]} -> Error: ${expected[i] - output[i]}")
            }

            model.compile(expected, input)

            // AFTER BACKPROPAGATION
            output = model.predict(input)
            error = model.lossFunction.totalLoss(output, expected)
            println("Mean squared error: $error")
            for (i in expected.indices) {
                println("Output: ${output[i]} : Expected: ${expected[i]} -> Error: ${expected[i] - output[i]}")
            }

            model.save("model.txt")
        }


        fun testBinaryClassification() {
            val data = DataLoader.loadIonosphere()

//            for (i in 0..<data.size()) {
//                println("Input: ${data.get(i).first.joinToString(", ")} -> Output: ${data.get(i).second}")
//            }

            val model = NeuralNetwork(learningRate = 0.01, lossFunction = BinaryCrossEntropy)
            model.addLayer(Layer(33, useBias = false))
            model.addLayer(Layer(4, ReLU, useBias = false))
            model.addLayer(Layer(4, ReLU, useBias = false))
            model.addLayer(Layer(1, Sigmoid, useBias = false))
            model.initialize()

            for (epoch in 0..<10) {
                println("Epoch: $epoch")
                for (index in 0..<data.size()) {
                    val input = data.get(index).first
                    val target = doubleArrayOf(data.get(index).second)

                    model.compile(target, input)
                }
            }

            model.save("model.txt")

            for (index in 0..<data.size()) {
                val input = data.get(index).first
                val target = doubleArrayOf(data.get(index).second)

                val output = model.predict(input)
                println(
                    "Output: ${Math.round(output[0])} : Expected: ${target[0]} -> Error: ${
                        target[0] - Math.round(
                            output[0]
                        )
                    }"
                )
            }
        }
    }
}