fun main() {
//    Tester.testBackpropagation()
    Tester.testBinaryClassification()
}




fun train(model: NeuralNetwork) {
    val environment = Environment()

    var epsilon = 1.0
    val episode = 100
    val step = 400

    // get current state
    var state = environment.getState()

    for (i in 0..<episode) {
        var totalReward = 0
        for (j in 0..<step) {

            val action = pickAction(epsilon, model, state)
            environment.action(action)
            val reward = environment.getReward()
            totalReward += reward

            model.compile(doubleArrayOf(reward.toDouble()), state)
        }
        println("Episode: $i")

        // Decrease epsilon
        epsilon = 0.1.coerceAtLeast(epsilon * 0.99)


        println("--------------------")
        println("Episode: $i")
        println("Total reward: $totalReward")
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


