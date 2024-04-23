fun main() {
    val model = NeuralNetwork(learningRate = 0.01)
    model.addLayer(Layer(5, Sigmoid, false))
    model.addLayer(Layer(10, Sigmoid, false))
    model.addLayer(Layer(10, Sigmoid, false))
    model.addLayer(Layer(10, Sigmoid, false))
    model.addLayer(Layer(10, Sigmoid, false))
    model.addLayer(Layer(10, Sigmoid, false))
    model.addLayer(Layer(5, Sigmoid, false))
    model.initialize()
    model.save("model.txt")


    val input = doubleArrayOf(1.5, 2.0, 3.0, 4.0, 5.0)
    val expected = doubleArrayOf(0.5, 0.2, 1.0, 0.4, 0.13)

    // BEFORE BACKPROPAGATION
    var output = model.predict(input)
    var error = model.totalSquaredError(output, expected)
    for (i in expected.indices) {
        println("Input: ${input[i]} -> Output: ${output[i]} : Expected: ${expected[i]} -> Error: $error")
    }


    model.backpropagation(expected, input)

    // AFTER BACKPROPAGATION
    //println(model)
    output = model.predict(input)
    error = model.totalSquaredError(output, expected)
    for (i in expected.indices) {
        println("Input: ${input[i]} -> Output: ${output[i]} : Expected: ${expected[i]} -> Error: $error")
    }


    //train(model)
}


fun train(model: NeuralNetwork) {
    val environment = Environment()

    var epsilon = 1.0
    val episode = 100
    val step = 400

    // get current state
    val state = environment.getState()

    for (i in 0 until episode) {
        for (j in 0 until step) {

            val action = pickAction(epsilon, model, state)
            environment.action(action)
            val reward = environment.getReward()
            val nextState = environment.getState()

            // Masque pour l'erreur ?
            var mask = DoubleArray(4) { 0.0 }
            mask[action] = 1.0

            // Insert les information dans un batch

        }

        // Decrease epsilon
        epsilon = Math.max(0.1, epsilon * 0.99)

//        println("--------------------")
//        println("Reward ")
//        println("episode: $i")
    }
}


fun pickAction(epsilon: Double, model: NeuralNetwork, currentState: DoubleArray): Int {
    if (Math.random() < epsilon) {
        return (0..3).random()
    } else {
        val result = model.predict(currentState)
        return result.indexOfFirst {
            it == result.max()
        }
    }
}

fun trainModel() {
}

