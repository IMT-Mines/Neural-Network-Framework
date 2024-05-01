package main.kotlin.network

import java.io.File

class NeuralNetwork(private var learningRate: Double, var lossFunction: LossFunction = SquaredError) {

    private val layers = mutableListOf<Layer>()

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
     * TODO COmprendre pourquoi le softmax ne fonctionne pas
     */
    fun stochasticGradientDescent(targets: DoubleArray) {

        val outputs = layers.last().getDerivativeOfEachNeuron()

        for (neuronIndex in layers.last().neurons.indices) {
            val neuron = layers.last().neurons[neuronIndex]
            for (weightIndex in neuron.weights.indices) {
                val outputError = this.lossFunction.derivative(neuron.output, targets[neuronIndex])

                neuron.delta = outputError * outputs[neuronIndex]
                //println("Partial error: $outputError, Derivative: ${outputs[neuronIndex]}")

                val nextLayerNeuron = layers[layers.size - 2].neurons[weightIndex]
                neuron.weights[weightIndex] -= learningRate * neuron.delta * nextLayerNeuron.output
            }
        }

        // TODO ENLEVER LE REVERSED
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
}