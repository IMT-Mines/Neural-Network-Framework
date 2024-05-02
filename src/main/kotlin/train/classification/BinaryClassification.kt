package main.kotlin.train.classification

import main.kotlin.charts.Chart
import main.kotlin.debug.DebugTools
import main.kotlin.network.*
import main.kotlin.train.Data
import main.kotlin.train.DataLoader
import org.jetbrains.kotlinx.kandy.util.color.Color

class BinaryClassification {

    private lateinit var debugTools: DebugTools

    /**
     * This function is used to classify the sonar dataset
     * If the label is "R" then the output is 1.0
     * If the label is "M" then the output is 0.0
     * The dataset has 60 features and 208 instances
     */
    fun sonarClassification() {
        // Load the data
        val data = DataLoader.loadSonar()
        val (train, test) = data.split(0.8)

        // Create the model
        val model = NeuralNetwork(learningRate = 0.001, lossFunction = BinaryCrossEntropy)
        model.addLayer(Layer(60))
        model.addLayer(Layer(60, ReLU))
        model.addLayer(Layer(10, ReLU))
        model.addLayer(Layer(1, Sigmoid))
        model.initialize()

        train(model, 1000, train)

        model.save("src/main/resources/sonarModel.txt")

        test(model, test)
    }

    /**
     * This function is used to classify the ionosphere dataset
     * If the label is "g" then the output is 1.0
     * If the label is "b" then the output is 0.0
     * The dataset has 33 features and 351 instances
     */
    fun ionosphereClassification() {
        // Load the data
        val data = DataLoader.loadIonosphere()
        val (train, test) = data.split(0.8)

        // Create the model
        val model = NeuralNetwork(learningRate = 0.001, lossFunction = BinaryCrossEntropy)
        model.addLayer(Layer(33))
        model.addLayer(Layer(4, ReLU))
        model.addLayer(Layer(4, ReLU))
        model.addLayer(Layer(1, Sigmoid))
        model.initialize()
        debugTools = DebugTools(model)

        train(model, 1000, train)

        model.save("src/main/resources/ionosphereModel.txt")

        debugTools.printWeightsOfNeuralNetwork()

        test(model, test)
    }

    private fun train(model: NeuralNetwork, epochs: Int, data: Data) {
        println("\n======================= TRAINING =======================\n")
        val lossChart: MutableMap<Int, Double> = mutableMapOf()
        val accuracyChart: MutableMap<Int, Double> = mutableMapOf()
        for (epoch in 1..epochs) {
            debugTools.archiveWeights()
            val accuracy = DoubleArray(data.size())
            var totalLoss = 0.0
            for (index in 0..<data.size()) {
                val sample = data.get(index)
                val inputs = sample.first
                val target = sample.second
                val outputs = model.predict(inputs)

                model.stochasticGradientDescent(target)

                if (Math.round(outputs[0]).toDouble() == target[0]) {
                    accuracy[index] = 1.0
                }
                totalLoss += model.lossFunction.loss(outputs, target)
            }
            accuracyChart[epoch] = accuracy.average()
            lossChart[epoch] = totalLoss / data.size()
            println(
                "Epoch: %d / %d | Training Loss: %10.4f | Accuracy: %10.2f".format(
                    epoch,
                    epochs,
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
        var accuracy = 0.0
        for (index in 0..<data.size()) {
            val sample = data.get(index)
            val inputs = sample.first
            val target = sample.second
            val output = model.predict(inputs)
            if (Math.round(output[0]).toDouble() == target[0]) {
                accuracy += 1.0
            }
            println(
                "Output: %10f | Expected: %10.0f | Error: %10.0f".format(
                    output[0],
                    target[0],
                    target[0] - Math.round(output[0])
                )
            )
        }
        println("\n=> On the test set, the model has an accuracy of ${accuracy / data.size()}")
    }
}