import kotlin.math.max

fun main() {
    val inputLayer = Layer(nbInputs = 1, nbNeurons = 3, activationFunction = ReLU)
    val hiddenLayer = Layer(nbInputs = 3, nbNeurons = 3, activationFunction = ReLU)
    val outputLayer = Layer(nbInputs = 3, nbNeurons = 1, activationFunction = Linear)

    val model = NeuralNetwork(listOf(inputLayer, hiddenLayer, outputLayer))
    train(model)
}


fun train(model: NeuralNetwork) {
    val map = IntArray(10) { 0 }

    var epsilon = 1.0
    val episode = 100000
    val step = 400

    for (i in 0 until episode) {
        for (j in 0 until step) {

            val action = pickAction(epsilon)
            model.predict(doubleArrayOf(1.0))
            // Effectuer l'action et obtenir la récompense
            // Récupérer le nouvel état

            // Masque pour l'erreur ?

            // Insert les information dans un batch

        }

        // Decrease epsilon
        epsilon = max(0.1, epsilon*0.99)

//        println("--------------------")
//        println("Reward ")
//        println("episode: $i")
    }
}

fun pickAction(epsilon: Double) {

    if (Math.random() < epsilon) {
        // Exploration
    } else {
        // Exploitation
    }
}


