package main.kotlin.network

import main.kotlin.charts.Chart
import main.kotlin.debug.DebugTools
import main.kotlin.train.Data
import org.jetbrains.kotlinx.kandy.util.color.Color
import java.io.File

class NeuralNetwork(private var learningRate: Double, var lossFunction: LossFunction = SquaredError) {

    val layers = mutableListOf<Layer>()

    fun predict(inputs: DoubleArray): DoubleArray {
        for (i in 0..<layers.first().neurons.size) {
            layers.first().neurons[i].output = inputs[i]
        }
        var outputs = inputs
        for (i in 1..<layers.size) {
            outputs = layers[i].compute(outputs)
        }
        return outputs
    }

    fun initialize() {
        if (layers.size < 2) throw IllegalArgumentException("The number of layers must be greater than 1")
        layers.first().initialize()
        for (i in 1..<layers.size) {
            layers[i].initialize(layers[i - 1].neurons.size)
        }
    }

    /**
     * This function is used to train the neural network, it uses the backpropagation algorithm
     */
    fun stochasticGradientDescent(targets: DoubleArray) {
        val outputsLayerDerivative = layers.last().getDerivativeOfEachNeuron()
        for (neuronIndex in layers.last().neurons.indices) {
            val neuron = layers.last().neurons[neuronIndex]
            for (weightIndex in neuron.weights.indices) {
                val outputError = this.lossFunction.derivative(neuron.output, targets[neuronIndex])
                neuron.delta = outputError * outputsLayerDerivative[neuronIndex]
                val nextLayerNeuron = layers[layers.size - 2].neurons[weightIndex]
                neuron.weights[weightIndex] -= learningRate * neuron.delta * nextLayerNeuron.output
            }
        }

        val reversedLayers = layers.reversed()
        for (layerIndex in 1..<reversedLayers.size - 1) {
            val outputsDerivatives = reversedLayers[layerIndex].getDerivativeOfEachNeuron()
            for (neuronIndex in reversedLayers[layerIndex].neurons.indices) {
                val neuron = reversedLayers[layerIndex].neurons[neuronIndex]
                for (weightIndex in neuron.weights.indices) {
                    var partialError = 0.0
                    for (previousLayerNeuron in reversedLayers[layerIndex - 1].neurons.indices) {
                        val beforeNeuron = reversedLayers[layerIndex - 1].neurons[previousLayerNeuron]
                        partialError += beforeNeuron.weights[neuronIndex] * beforeNeuron.delta
                    }
                    neuron.delta = partialError * outputsDerivatives[neuronIndex]
                    val nextLayerNeuron = reversedLayers[layerIndex + 1].neurons[weightIndex]
                    neuron.weights[weightIndex] -= learningRate * neuron.delta * nextLayerNeuron.output
                }
            }
        }
    }

    fun fit(epochs: Int, data: Data, debug: Boolean = false) {
        val debugTools = DebugTools(this)
        println("\n======================= TRAINING =======================\n")
        val lossChart: MutableMap<Int, Double> = mutableMapOf()
        val accuracyChart: MutableMap<Int, Double> = mutableMapOf()
        for (epoch in 0..<epochs) {
            if (debug) debugTools.run { archiveWeights(); archiveDelta() }
            val accuracy = DoubleArray(data.size())
            var totalLoss = 0.0
            for (index in 0..<data.size()) {
                val sample = data.get(index)
                val (inputs, target) = sample
                val outputs = this.predict(inputs)
                this.stochasticGradientDescent(target)

                accuracy[index] = getAccuracy(outputs, target)

                totalLoss += this.lossFunction.loss(outputs, target)
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
        if (debug) {
            debugTools.printDeltas()
            debugTools.printWeights()
        }
        Chart.lineChart(accuracyChart, "Model accuracy", "Epoch", "Accuracy", Color.GREEN, "src/main/resources")
        Chart.lineChart(lossChart, "Model loss", "Epoch", "Loss", Color.BLUE, "src/main/resources")
    }

    fun test(data: Data) {
        println("\n======================= TESTING =======================\n")
        var accuracy = 0.0
        for (index in 0..<data.size()) {
            val sample = data.get(index)
            val inputs = sample.first
            val target = sample.second
            val outputs = this.predict(inputs)

            accuracy += getAccuracy(outputs, target)

            println(
                "Output: ${outputs.joinToString { "%.2f".format(it) }} | Target: ${
                    target.joinToString {
                        "%.2f".format(
                            it
                        )
                    }
                }"
            )
        }
        println("\nThe Accuracy on the test set is: ${accuracy / data.size()}")
    }

    fun addLayer(layer: Layer) {
        layers.add(layer)
    }

    fun load(path: String) {
        val file = File(path)
        val bufferedReader = file.bufferedReader()
        var line = bufferedReader.readLine()
        while (line != null) {
            val nbNeurons = line.toInt()
            val layer = Layer(nbNeurons)
            layer.initialize()
            for (i in 0..<nbNeurons) {
                line = bufferedReader.readLine()
                if (line.isEmpty()) continue
                val weights = line.split(" ").map { it.toDouble() }.toDoubleArray()
                layer.neurons[i].initialize(values = weights)
            }
            layers.add(layer)
            line = bufferedReader.readLine()
        }
        bufferedReader.close()
    }

    fun save(path: String) {
        val file = File(path)
        val bufferWriter = file.bufferedWriter()
        for (layer in layers) {
            bufferWriter.write("${layer.neurons.size}\n")
            for (neuron in layer.neurons) {
                val str = neuron.weights.joinToString(" ")
                bufferWriter.write(str)
                bufferWriter.write("\n")
            }
        }
        bufferWriter.close()
    }

    override fun toString(): String {
        val sb = StringBuilder()
        for (i in layers.indices) {
            sb.append("Layer $i\n")
            // name of the activation function
            sb.append("Activation function: ${layers[i].activationFunction}\n")
            sb.append(layers[i].toString())
            sb.append("\n")
        }
        return sb.toString()
    }

    private fun getAccuracy(outputs: DoubleArray, target: DoubleArray): Double {
        return if (outputs.size == 1) {
            when {
                outputs[0] >= 0.5 && target[0] == 1.0 -> 1.0
                outputs[0] < 0.5 && target[0] == 0.0 -> 1.0
                else -> 0.0
            }
        } else {
            if (outputs.withIndex().maxByOrNull { it.value }?.index == target.withIndex()
                    .maxByOrNull { it.value }?.index
            ) 1.0 else 0.0
        }
    }
}