package main.kotlin

import main.kotlin.train.classification.BinaryClassification
import main.kotlin.train.classification.MultiClassification
import main.kotlin.train.other.Tester


fun main() {
    val binaryClassification = BinaryClassification()
    binaryClassification.ionosphereClassification()

//    val multiClassification = MultiClassification()
//    multiClassification.seedsClassification()

//    val tester = Tester()
//    tester.testInitialization()
}

