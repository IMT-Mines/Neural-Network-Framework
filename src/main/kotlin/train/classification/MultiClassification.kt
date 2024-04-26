package main.kotlin.train.classification

import main.kotlin.network.*
import main.kotlin.train.DataLoader

class MultiClassification {

    /**
     * This function is used to classify the iris dataset
     * If the label is "Iris-setosa" then the output is 0.0
     * If the label is "Iris-versicolor" then the output is 1.0
     * If the label is "Iris-virginica" then the output is 2.0
     * The dataset has 4 features and 150 instances
     */
    fun irisClassification() {
        // Load the data
        val data = DataLoader.loadIris()

        // Create the model
        val model = NeuralNetwork(learningRate = 0.001, lossFunction = BinaryCrossEntropy)
        model.addLayer(Layer(4))
        model.addLayer(Layer(3, ReLU))
        model.addLayer(Layer(1, Sigmoid))
        model.initialize()

        // Train the model
        for (epoch in 0..<7) {
            println("Epoch: $epoch")

            for (index in 0..<data.size()) {
                val input = data.get(index).first
                val target = doubleArrayOf(data.get(index).second)

                model.compile(target, input)
            }
        }

        model.save("src/main/resources/irisModel.txt")
        for (index in 0..<data.size()) {
            val input = data.get(index).first
            val target = doubleArrayOf(data.get(index).second)

            val output = model.predict(input)
            println(
                "Output: ${Math.round(output[0])} : Expected: ${target[0]} -> Error: ${
                    target[0] - Math.round(
                        output[0]
                    )
                }"
            )
        }
    }
}