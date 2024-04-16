import java.io.File
import kotlin.math.pow

class NeuralNetwork(private var learningRate: Double) {

    private val layers = mutableListOf<Layer>()

    fun predict(inputs: DoubleArray): DoubleArray {
        var outputs = inputs
        for (layer in layers) {
            outputs = layer.compute(outputs)
        }
        return outputs
    }

    fun addLayer(layer: Layer) {
        layers.add(layer)
    }

    fun meanSquaredError(predicted: DoubleArray, expected: DoubleArray): Double {
        var error = 0.0
        for (i in predicted.indices) {
            error += (expected[i] - predicted[i]).pow(2.0)
        }
        return error
    }

    /**
     * Compute the gradiant of the weights
     * w0' = w0 - r * a1 * 2 (a0 - y)
     * w1' = w1 - r * a1 * w0 * 2 (a0 - y)
     *
     * where:
     * - wx is the weight of a neuron
     * - r is the learning rate
     * - a0 is the output of the neuron
     * - a1 is the input of the neuron
     * - y is the expected output
     */
    private fun gradiant(
        output: Double,
        currentWeight: Double,
        expected: Double,
        weightsPrev: Double,
        input: Double
    ): Double {
        println("w0 = $currentWeight - r * $input * $weightsPrev * 2($output - $expected)")
        return currentWeight - learningRate * input * weightsPrev * 2 * (output - expected)
    }


    fun backpropagation() {
        repeat(5) {
            // var weightsPrev = 1.0
            // reset neurons backpropagation
            for (layer in layers) {
                for (neuron in layer.neurons) {
                    neuron.backpropagation = 1.0
                }
            }

            val output = predict(doubleArrayOf(1.5))
            for (layer in layers.reversed()) {
                for (neuron in layer.neurons) {
                    for (i in neuron.weights.indices) {
                        if (layer != layers.last()) {
                            neuron.weights[i] = gradiant(output[0], neuron.weights[i], 0.5, 1.0, 1.5)
                        } else {
                            neuron.weights[i] = gradiant(output[0], neuron.weights[i], 0.5, neuron.backpropagation, 1.5)
                        }
                        neuron.backpropagation *= neuron.weights[i]
                    }
                }
            }
            println("==========================")
        }
    }

    fun load(path: String) {
        val file = File(path)
        val bufferedReader = file.bufferedReader()
        var line = bufferedReader.readLine()
        while (line != null) {
            val (nbInputs, nbNeurons) = line.split(" ").map { it.toInt() }
            val layer = Layer(nbInputs, nbNeurons, ReLU, false)
            line = bufferedReader.readLine()
            val lineWeights = line.split(" ").subList(0, nbNeurons * nbInputs)
            val weights = lineWeights.map { it.toDouble() }.toDoubleArray()
            for (neuron in layer.neurons) {
                neuron.weights = weights
            }
            layers.add(layer)
            line = bufferedReader.readLine()
        }
    }


    fun save(path: String) {
        val file = File(path)
        val bufferWriter = file.bufferedWriter()
        for (layer in layers.reversed()) {
            bufferWriter.write("${layer.neurons.size} ${layer.neurons[0].weights.size}\n")
            for (neuron in layer.neurons) {
                for (weight in neuron.weights) {
                    bufferWriter.write("$weight ")
                }
            }
            bufferWriter.write("\n")
        }
        bufferWriter.close()
    }

    override fun toString(): String {
        val sb = StringBuilder()
        for (layer in layers) {
            sb.append("Layer\n")
            sb.append(layer.toString())
            sb.append("\n")
        }
        return sb.toString()
    }
}