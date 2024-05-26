package main.kotlin

import main.kotlin.train.classification.MultiClassification
import main.kotlin.train.deepReinforcement.ReinforcementLearning

//import DeepQLearning
//import main.kotlin.train.deepReinforcement.main.kotlin.train.deepReinforcement.Environment


fun main() {
//    val binaryClassification = BinaryClassification()
//    binaryClassification.ionosphereClassification()

//    val multiClassification = MultiClassification()
//    multiClassification.digitsClassification()

    val reinforcementLearning = ReinforcementLearning()
    reinforcementLearning.foodGameTraining()
    reinforcementLearning.test()

//    val tester = Tester()
//    tester.testInitialization()
}

