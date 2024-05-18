package main.kotlin

import main.kotlin.train.classification.BinaryClassification

//import DeepQLearning
//import main.kotlin.train.deepReinforcement.Environment


fun main() {
//    val binaryClassification = BinaryClassification()
//    binaryClassification.ionosphereClassification()

    val multiClassification = MultiClassification()
    multiClassification.digitsClassification()

//    val environment = Environment(3)
//    val learningRate = 0.01
//    val deepQLearning = DeepQLearning(environment, learningRate)
//    deepQLearning.train(episodes = 150, epsilon = 1.0, replayBufferSize = 10000)


//    val tester = Tester()
//    tester.testInitialization()
}

