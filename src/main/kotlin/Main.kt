package main.kotlin

import main.kotlin.charts.Chart
import main.kotlin.network.CategoricalCrossEntropy
import main.kotlin.network.MeanSquaredError
import main.kotlin.network.Sigmoid
import main.kotlin.network.Softmax
import main.kotlin.train.classification.BinaryClassification
import main.kotlin.train.classification.MultiClassification
import main.kotlin.train.other.Tester


fun main() {
//    val binaryClassification = BinaryClassification()
//    binaryClassification.ionosphereClassification()

    val multiClassification = MultiClassification()
    multiClassification.irisClassification()

//    val tester = Tester()
//    tester.testBackpropagation()
}

