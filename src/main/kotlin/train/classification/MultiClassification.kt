package main.kotlin.train.classification

import main.kotlin.network.CategoricalCrossEntropy
import main.kotlin.network.Layer
import main.kotlin.network.NeuralNetwork
import main.kotlin.network.ReLU
import main.kotlin.train.Data
import main.kotlin.train.DataLoader

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
        val model = NeuralNetwork(learningRate = 0.01, lossFunction = CategoricalCrossEntropy)
        model.addLayer(Layer(4))
        model.addLayer(Layer(3, ReLU))
        model.addLayer(Layer(3, ReLU)) // SoftMax
        model.initialize()

        train(model, 20, train)

        model.save("src/main/resources/irisModel.txt")

        test(model, test)
    }

    private fun train(model: NeuralNetwork, epochs: Int, data: Data) {
        println("\n======================= TRAINING =======================\n")
        for (epoch in 0..<epochs) {
            val accuracy = DoubleArray(data.size())
            var totalLoss = 0.0
            for (index in 0..<data.size()) {
                val sample = data.get(index)
                val inputs = sample.first
                val target = sample.second
                model.compile(target)

                val outputs = model.predict(inputs)
//                println(
//                    "Output: %s | Expected: %s".format(
//                        outputs.joinToString { it.toString() },
//                        target.joinToString { it.toString() },
//                    )
//                )

                val maxOutputIndex = outputs.withIndex().maxByOrNull { it.value }?.index
                val targetIndex = target.withIndex().maxByOrNull { it.value }?.index
                if (maxOutputIndex == targetIndex) {
                    accuracy[index] = 1.0
                }
                totalLoss += model.lossFunction.totalLoss(outputs, target)
            }
            println(
                "Epoch: %d | Training Loss: %10.4f | Accuracy: %10.2f".format(
                    epoch,
                    totalLoss / data.size(),
                    accuracy.average()
                )
            )
        }
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