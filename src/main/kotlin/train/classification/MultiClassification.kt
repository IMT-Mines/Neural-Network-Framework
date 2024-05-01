package main.kotlin.train.classification

import main.kotlin.charts.Chart
import main.kotlin.network.*
import main.kotlin.train.Data
import main.kotlin.train.DataLoader
import org.jetbrains.kotlinx.kandy.util.color.Color

class MultiClassification {

    /**
     * This function is used to classify the iris dataset
     * If the label is "Iris-setosa" then the label is [1, 0, 0]
     * If the label is "Iris-versicolor" then the label is [0, 1, 0]
     * If the label is "Iris-virginica" then the label is [0, 0, 1]
     * The dataset has 4 features and 150 instances
     */
    fun irisClassification() {
        // Load the data
        val data = DataLoader.loadIris()
        val (train, test) = data.split(0.8)

        // Create the model
        val model = NeuralNetwork(learningRate = 0.001, lossFunction = CategoricalCrossEntropy)
        model.addLayer(Layer(4))
        model.addLayer(Layer(3, ReLU))
        model.addLayer(Layer(3, ReLU))
        model.addLayer(Layer(3, Softmax))
        model.initialize()

        train(model, 1000, train)

        model.save("src/main/resources/irisModel.txt")

        test(model, test)
    }

    private fun train(model: NeuralNetwork, epochs: Int, data: Data) {
        println("\n======================= TRAINING =======================\n")
        val lossChart: MutableMap<Int, Double> = mutableMapOf()
        val accuracyChart: MutableMap<Int, Double> = mutableMapOf()
        for (epoch in 0..<epochs) {
            val accuracy = DoubleArray(data.size())
            var totalLoss = 0.0
            for (index in 0..<data.size()) {
                val sample = data.get(index)
                val inputs = sample.first
                val target = sample.second
                model.stochasticGradientDescent(target)

                val outputs = model.predict(inputs)

                val maxOutputIndex = outputs.withIndex().maxByOrNull { it.value }?.index
                val targetIndex = target.withIndex().maxByOrNull { it.value }?.index
                if (maxOutputIndex == targetIndex) {
                    accuracy[index] = 1.0
                }
                totalLoss += model.lossFunction.loss(outputs, target)
            }
            accuracyChart[epoch] = accuracy.average()
            lossChart[epoch] = totalLoss / data.size()
            println(
                "Epoch: %d | Training Loss: %10.4f | Accuracy: %10.2f".format(
                    epoch,
                    totalLoss / data.size(),
                    accuracy.average()
                )
            )
        }
        Chart.lineChart(accuracyChart, "Model accuracy", "Epoch", "Accuracy", Color.GREEN)
        Chart.lineChart(lossChart, "Model loss", "Epoch", "Loss")
    }

    private fun test(model: NeuralNetwork, data: Data) {
        println("\n======================= TESTING =======================\n")
        for (index in 0..<data.size()) {
            val sample = data.get(index)
            val inputs = sample.first
            val target = sample.second
            val output = model.predict(inputs)

            println(
                "Output: %s | Expected: %s".format(
                    output.joinToString { it.toString() },
                    target.joinToString { it.toString() },
                )
            )
        }
    }
}