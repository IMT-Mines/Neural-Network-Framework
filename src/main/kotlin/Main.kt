package main.kotlin

import main.kotlin.train.classification.MultiClassification


fun main() {
//    val binaryClassification = BinaryClassification()
//    binaryClassification.ionosphereClassification()

    val multiClassification = MultiClassification()
    multiClassification.irisClassification()

//    val tester = Tester()
//    tester.testBackpropagation()
}

