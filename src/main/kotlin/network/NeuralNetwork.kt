package main.kotlin.network

import main.kotlin.charts.Chart
import main.kotlin.train.Data
import main.kotlin.utils.DebugTools
import org.jetbrains.kotlinx.kandy.util.color.Color
import java.io.File
import java.util.*
import java.util.concurrent.CountDownLatch
import java.util.concurrent.Executors


class NeuralNetwork(private var learningRate: Double, var loss: Loss = SquaredError) {

    val layers = mutableListOf<Layer>()

    fun predict(inputs: DoubleArray): DoubleArray {
        for (i in 0..<layers.first().nbNeurons) {
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
            layers[i].initialize(layers[i - 1].nbNeurons)
        }
    }

    /**
     * This function is used to train the neural network, it uses the backpropagation algorithm
     */
    private fun gradientDescent(loss: DoubleArray) {
        val outputsLayerDerivative = layers.last().getDerivativeOfEachNeuron()
        for (neuronIndex in layers.last().neurons.indices) {
            val neuron = layers.last().neurons[neuronIndex]
            for (weightIndex in neuron.weights.indices) {
//                val outputError = this.loss.derivative(neuron.output, targets[neuronIndex])
                neuron.delta = loss[neuronIndex] * outputsLayerDerivative[neuronIndex]
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

    fun fit(epochs: Int, data: Data, batchSize: Int = 1, debug: Boolean = false) {
        if (data.size() % batchSize != 0) throw IllegalArgumentException("The batch size must be a multiple of the data size")

        val debugTools = DebugTools(this)
        println("\n======================= TRAINING =======================\n")
        val lossChart: MutableMap<Int, Double> = mutableMapOf()
        val accuracyChart: MutableMap<Int, Double> = mutableMapOf()

        val batchCount = data.size() / batchSize
        val threadPool = Executors.newFixedThreadPool(10)

        for (epoch in 0..<epochs) {
            if (debug) debugTools.run { archiveWeights(); archiveDelta() }
            val accuracy = DoubleArray(data.size())
            data.shuffle()
            var totalLoss = 0.0

            val finalLoss = this.loss

            val latch = CountDownLatch(batchCount)

            for (batchIndex in 0 until batchCount) {

                val lossDerivationSum = DoubleArray(layers.last().nbNeurons)

                threadPool.submit {
                    try {
                        for (index in 0 until batchSize) {
                            val sample = data.get(batchIndex * batchSize + index)
                            val (inputs, target) = sample
                            val outputs = predict(inputs)

                            val loss = DoubleArray(outputs.size)
                            for (i in outputs.indices) {
                                loss[i] = finalLoss.derivative(outputs[i], target[i])
                            }
                            lossDerivationSum.mapIndexed { i, value -> value + loss[i] }
                        }
                    } finally {
                        latch.countDown()
                    }
                }

                this.gradientDescent(lossDerivationSum.map { it / batchSize }.toDoubleArray())
//                accuracy[batchIndex * batchSize + index] = getAccuracy(outputs, target)
//                totalLoss += this.loss.averageLoss(outputs, target)
            }

            latch.await()

            accuracyChart[epoch] = accuracy.average()
//            lossChart[epoch] = totalLoss / data.size()
            println(
                "Epoch: %d | Training Loss: %10.4f | Accuracy: %10.2f".format(
                    epoch,
                    totalLoss / data.size(),
                    accuracy.average()
                )
            )
        }

        threadPool.shutdown()

        if (debug) debugTools.run { debugTools.printDeltas(); debugTools.printWeights() }
        Chart.lineChart(accuracyChart, "Model accuracy", "Epoch", "Accuracy", Color.GREEN, "src/main/resources/plots")
        Chart.lineChart(lossChart, "Model loss", "Epoch", "Loss", Color.BLUE, "src/main/resources/plots")
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
            val activationFunctionName = bufferedReader.readLine()
            val activationFunction = when (activationFunctionName) {
                "ReLU" -> ReLU
                "Sigmoid" -> Sigmoid
                "Softmax" -> Softmax
                "Tanh" -> Tanh
                "LeakyReLU" -> LeakyReLU
                else -> Linear
            }
            val layer = Layer(nbNeurons, activationFunction)
            layer.bias = bufferedReader.readLine().toDouble()
            for (i in 0..<nbNeurons) {
                line = bufferedReader.readLine()
                if (line.isEmpty()) continue
                val weights = line.split(" ").map { it.toDouble() }.toDoubleArray()
                layer.neurons[i].weights = weights
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
            bufferWriter.write("${layer.nbNeurons}\n")
            layer.activation::class.simpleName?.let { bufferWriter.write("$it\n") }
            bufferWriter.write("${layer.bias}\n")
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
        sb.append("_________________________________________________________________")
        sb.append("\nLayer                     Output Shape          Weights")
        sb.append("\n=================================================================")
        for (i in layers.indices) {
            val nbLayerWeights = layers[i].neurons.first().weights.size * layers[i].nbNeurons
            sb.append(
                "\nLayer_%d (Dense)           (%s, %d)              %d"
                    .format(
                        i,
                        layers[i].activation::class.simpleName,
                        layers[i].nbNeurons,
                        nbLayerWeights
                    )
            )
            sb.append("\n_________________________________________________________________")
        }
        return sb.toString()
    }

    private fun averageColumn(matrix: MutableList<DoubleArray>): DoubleArray {
        val average = DoubleArray(matrix.first().size)
        for (i in matrix.indices) {
            for (j in matrix[i].indices) {
                average[j] += matrix[i][j]
            }
        }
        return average.map { it / matrix.size }.toDoubleArray()
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