package main.kotlin

import main.kotlin.network.MeanSquaredError
import main.kotlin.network.Softmax
import main.kotlin.train.classification.BinaryClassification
import main.kotlin.train.classification.MultiClassification


fun main() {
    val binaryClassification = BinaryClassification()
    binaryClassification.sonarClassification()

//    val multiClassification = MultiClassification()
//    multiClassification.irisClassification()

//    // Test de l'activation Softmax
//    val inputArray = doubleArrayOf(0.4, 0.4, 0.2)
//    println("Input array: ${inputArray.joinToString()}")
//
//    val softmax = Softmax.activate(inputArray)
//    println("Softmax activation: ${softmax.joinToString()}")
//
//    // Test de la dérivée de Softmax
//    val inputArrayDerivative = doubleArrayOf(0.4, 0.4, 0.2)
//    println("Input array for derivative: ${inputArrayDerivative.joinToString()}")
//
//    val softmaxDerivative = Softmax.derivative(inputArrayDerivative)
//    println("Softmax derivative: ${softmaxDerivative.joinToString()}")
//
//    val totalLoss= MeanSquaredError.loss(inputArray, doubleArrayOf(0.0, 1.0, 0.0))
//    println("Total loss: $totalLoss")

//    val tester = Tester()
//    tester.testBackpropagation()
}

