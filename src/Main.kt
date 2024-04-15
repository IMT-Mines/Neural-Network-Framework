fun main() {
    val model = NeuralNetwork(learningRate = 0.01)
    model.addLayer(Layer(nbInputs = 2, nbNeurons = 10, activationFunction = ReLU))
    model.addLayer(Layer(nbInputs = 10, nbNeurons = 4, activationFunction = Linear))


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

