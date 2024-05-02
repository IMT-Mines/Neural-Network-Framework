package main.kotlin

import main.kotlin.charts.Chart
import main.kotlin.network.CategoricalCrossEntropy
import main.kotlin.network.MeanSquaredError
import main.kotlin.network.Sigmoid
import main.kotlin.network.Softmax
import main.kotlin.train.classification.BinaryClassification
import main.kotlin.train.classification.MultiClassification


fun main() {
    val binaryClassification = BinaryClassification()
    binaryClassification.ionosphereClassification()

//    val multiClassification = MultiClassification()
//    multiClassification.irisClassification()

    // Test de l'activation Softmax
    val inputArray = doubleArrayOf(0.5, 0.0, 0.5)
    println("Input array: ${inputArray.joinToString()}")

    val softmax = Softmax.activate(inputArray)
    println("Softmax activation: ${softmax.joinToString()}")

    val softmaxDerivative = Softmax.derivative(inputArray)
    println("Softmax derivative: ${softmaxDerivative.joinToString()}")

    val totalLoss= CategoricalCrossEntropy.loss(inputArray, doubleArrayOf(0.0, 1.0, 0.0))
    println("Total loss: $totalLoss")


    println("\n\n ====== \n\n")



//    val tester = Tester()
//    tester.testBackpropagation()
}

