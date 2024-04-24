import java.io.File
import kotlin.math.pow

class NeuralNetwork(private var learningRate: Double) {

    private val layers = mutableListOf<Layer>()

    fun predict(inputs: DoubleArray): DoubleArray {
        for (i in 0..<layers.first().neurons.size) {
            layers.first().neurons[i].value = inputs[i]
        }
        var outputs = inputs
        for (i in 1..<layers.size) {
            outputs = layers[i].compute(outputs)
        }
        return outputs
    }

    fun addLayer(layer: Layer) {
        layers.add(layer)
    }

    fun initialize() {
        if (layers.size < 2) throw IllegalArgumentException("The number of layers must be greater than 1")
        layers.first().initialize()
        for (i in 1..<layers.size) {
            layers[i].initialize(layers[i - 1].neurons.size)
        }
    }

    fun meanSquaredError(actual: DoubleArray, expected: DoubleArray): Double {
        var error = 0.0
        for (i in actual.indices) {
            error += (expected[i] - actual[i]).pow(2)
        }
        return 1.0 / actual.size * error
    }

    fun optimize(target: DoubleArray, input: DoubleArray) {
        var error = 1.0
        var iteration = 0
        while (error > 0.0001) {
            for (neuronIndex in layers.last().neurons.indices) {
                val neuron = layers.last().neurons[neuronIndex]
                for (weightIndex in neuron.weights.indices) {
                    val partialError = neuron.value - target[neuronIndex]
                    val partialDerivative = neuron.activationFunction.derivative(neuron.value)
                    neuron.backPropagation = partialError * partialDerivative

                    val nextNeuron = layers[layers.size - 2].neurons[neuronIndex]
                    neuron.weights[weightIndex] -= learningRate * nextNeuron.value * neuron.backPropagation
                }
            }

            val reversedLayers = layers.reversed()
            for (layerIndex in 1..<reversedLayers.size - 1) {
                for (neuronIndex in reversedLayers[layerIndex].neurons.indices) {
                    val neuron = reversedLayers[layerIndex].neurons[neuronIndex]
                    for (weightIndex in neuron.weights.indices) {
                        var partialError = 0.0
                        for (beforeNeuronIndex in reversedLayers[layerIndex - 1].neurons.indices) {
                            val beforeNeuron = reversedLayers[layerIndex - 1].neurons[beforeNeuronIndex]
                            partialError += beforeNeuron.weights[neuronIndex] * beforeNeuron.backPropagation
                        }
                        val partialDerivative = neuron.activationFunction.derivative(neuron.value)
                        neuron.backPropagation = partialError * partialDerivative

                        val nextNeuron = reversedLayers[layerIndex + 1].neurons[weightIndex]
                        neuron.weights[weightIndex] -= learningRate * neuron.backPropagation * nextNeuron.value
                    }
                }
            }
            iteration++
            val predict = predict(input)
            error = meanSquaredError(predict, target)
        }
        println("Finished in $iteration iterations with error $error")
    }

    fun load(path: String) {
        val file = File(path)
        val bufferedReader = file.bufferedReader()
        var line = bufferedReader.readLine()
        while (line != null) {
            val (nbNeurons) = line.split(" ").map { it.toInt() }
            val layer = Layer(nbNeurons, Linear, false)
            layer.initialize()
            for (i in 0..<nbNeurons) {
                line = bufferedReader.readLine()
                if (line.isEmpty()) break
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
            sb.append(layers[i].toString())
            sb.append("\n")
        }
        return sb.toString()
    }
}

