package main.kotlin.debug

import main.kotlin.charts.Chart
import main.kotlin.network.NeuralNetwork
import main.kotlin.network.Neuron

class DebugTools(private val model: NeuralNetwork){

    private var weightsHistory: MutableMap<Neuron, MutableList<DoubleArray>> = mutableMapOf()


    fun archiveWeights() {
        for (layer in model.layers) {
            for (neuron in layer.neurons) {
                if (weightsHistory[neuron] == null) {
                    weightsHistory[neuron] = mutableListOf()
                }
                weightsHistory[neuron]?.add(neuron.weights.copyOf())
            }
        }
    }


    /**
     * Print the weights of the neural network, to debug the training process, you must add archiveWeights method in the training loop in every epoch
     * With this function you can see if the weights are changing
     */
    fun printWeightsOfNeuralNetwork() {
        for (layerIndex in 1..<model.layers.size) {
            for (neuronIndex in model.layers[layerIndex].neurons.indices) {

                val weightList: MutableList<MutableMap<Int, Double>> = mutableListOf()

                for (weightIndex in 0..<model.layers[layerIndex].neurons[neuronIndex].weights.size) {
                    weightList.add(mutableMapOf())
                    for (i in 0..< weightsHistory[model.layers[layerIndex].neurons[neuronIndex]]!!.size) {
                        weightList[weightIndex][i] = weightsHistory[model.layers[layerIndex].neurons[neuronIndex]]!![i][weightIndex]
                    }
                }

                Chart.multiLineChart(
                    weightList,
                    "Weights layer $layerIndex neuron $neuronIndex", "Epoch", "Weight", "weight"
                )
            }
        }
    }
}