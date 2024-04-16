fun main() {
    val model = NeuralNetwork(learningRate = 0.1)
//    model.addLayer(Layer(1, 1, ReLU))
//    model.save("model.txt")
//


    model.load("model.txt")

    val input = doubleArrayOf(1.5)
    val expected = doubleArrayOf(0.5)
    var output = model.predict(input)
    var error = model.meanSquaredError(output, expected)
    println("Input: ${input[0]} -> Output: ${output[0]} : Expected: ${expected[0]} -> Error: $error")


    model.backpropagation()
    println(model)
    output = model.predict(input)
    error = model.meanSquaredError(output, expected)
    println("Input: ${input[0]} -> Output: ${output[0]} : Expected: ${expected[0]} -> Error: $error")


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

