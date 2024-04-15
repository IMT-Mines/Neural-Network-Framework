import java.io.File
import kotlin.math.pow

class NeuralNetwork(private var learningRate: Double) {

    val layers = mutableListOf<Layer>()

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

    fun idk(input: Double, weight: Double, expected: Double): Double {
        val val1 = input * 2 * input
        val val2 = input * 2 * expected
        return weight - learningRate * (val1 * weight - val2)
    }

    fun backpropagation() {
        for (i in 0..<500) {
            layers[0].neurons[0].weights[0] = idk(1.5, layers[0].neurons[0].weights[0], 0.5)
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
            bufferWriter.write("${layer.neurons[0].weights.size} ${layer.neurons.size}\n")
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