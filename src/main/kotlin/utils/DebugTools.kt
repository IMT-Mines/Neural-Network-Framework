package main.kotlin.utils

import main.kotlin.charts.Chart
import main.kotlin.network.NeuralNetwork
import main.kotlin.network.Neuron
import org.jetbrains.kotlinx.kandy.util.color.Color
import java.io.File

class DebugTools(private val model: NeuralNetwork){

    private var weightsHistory: MutableMap<Neuron, MutableList<DoubleArray>> = mutableMapOf()
    private var deltaHistory: MutableMap<Neuron, MutableList<Double>> = mutableMapOf()
    private var biasHistory: MutableMap<Neuron, MutableList<Double>> = mutableMapOf()

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

    fun archiveDelta() {
        for (layer in model.layers) {
            for (neuron in layer.neurons) {
                if (deltaHistory[neuron] == null) {
                    deltaHistory[neuron] = mutableListOf()
                }
                deltaHistory[neuron]?.add( neuron.delta)
            }
        }
    }

    fun archiveBias() {
        for (layer in model.layers) {
            for (neuron in layer.neurons) {
                if (biasHistory[neuron] == null) {
                    biasHistory[neuron] = mutableListOf()
                }
                biasHistory[neuron]?.add( neuron.bias)
            }
        }
    }


    fun printDeltas() {
        val path = "src/main/resources/plots/debug/delta"
        val folder = File(path)
        folder.deleteRecursively()

        for (layerIndex in 1..<model.layers.size) {
            for (neuronIndex in model.layers[layerIndex].neurons.indices) {

                val deltaList: MutableMap<Int, Double> = mutableMapOf()

                for (i in 0..< deltaHistory[model.layers[layerIndex].neurons[neuronIndex]]!!.size) {
                    deltaList[i] = deltaHistory[model.layers[layerIndex].neurons[neuronIndex]]!![i]
                }

                Chart.lineChart(
                    deltaList,
                    "Layer-$layerIndex Neuron-$neuronIndex", "Epoch", "Delta", Color.ORANGE, path
                )
            }
        }
    }

    fun printBias() {
        val path = "src/main/resources/plots/debug/bias"
        val folder = File(path)
        folder.deleteRecursively()

        for (layerIndex in 1..<model.layers.size) {
            for (neuronIndex in model.layers[layerIndex].neurons.indices) {

                val deltaList: MutableMap<Int, Double> = mutableMapOf()

                for (i in 0..< biasHistory[model.layers[layerIndex].neurons[neuronIndex]]!!.size) {
                    deltaList[i] = biasHistory[model.layers[layerIndex].neurons[neuronIndex]]!![i]
                }

                Chart.lineChart(
                    deltaList,
                    "Layer-$layerIndex Neuron-$neuronIndex", "Epoch", "Bias", Color.RED, path
                )
            }
        }
    }

    /**
     * Print the weights of the neural network, to debug the training process, you must add archiveWeights method in the training loop in every epoch
     * With this function you can see if the weights are changing
     */
    fun printWeights() {
        val path = "src/main/resources/plots/debug/weights"
        val folder = File(path)
        folder.deleteRecursively()

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
                    "Layer-$layerIndex Neuron-$neuronIndex", "Epoch", "Weight", "weight", path
                )
            }
        }
    }
}