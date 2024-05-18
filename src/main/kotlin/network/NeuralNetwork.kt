package main.kotlin.network

import main.kotlin.charts.Chart
import main.kotlin.train.Data
import main.kotlin.utils.DebugTools
import main.kotlin.utils.Utils.Companion.awaitFutures
import org.jetbrains.kotlinx.kandy.util.color.Color
import java.io.File
import java.util.concurrent.CompletableFuture
import java.util.concurrent.Executors

class NeuralNetwork(var loss: Loss = SquaredError, var optimizer: Optimizer = SGD(0.001)) {

    val layers = mutableListOf<Layer>()

    private val threadPool = Executors.newFixedThreadPool(512)

    private fun predict(inputs: DoubleArray): DoubleArray {
        for (i in 0..<layers.first().nbNeurons) {
            layers.first().neurons[i].output = inputs[i]
        }
        var outputs = inputs
        for (i in 1..<layers.size) {
            outputs = layers[i].compute(outputs)
        }
        return outputs
    }

    private fun buildNeuralModel() {
        if (layers.size < 2) throw IllegalArgumentException("The number of layers must be greater than 1")
        layers.first().buildLayer()
        for (i in 1..<layers.size) {
            layers[i].buildLayer(layers[i - 1].nbNeurons)
        }
    }

    fun initialize() {
        buildNeuralModel()
        for (layerIndex in 1..<layers.size) {
            for (neuronIndex in layers[layerIndex].neurons.indices) {
                val fanIn = layers[layerIndex].neurons[neuronIndex].weights.size
                val fanOut = if (layerIndex == layers.size - 1) 0 else layers[layerIndex + 1].neurons.size
                val weights = layers[layerIndex].initialization.initialize(fanIn, fanOut)
                layers[layerIndex].neurons[neuronIndex].weights = weights
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

        for (epoch in 0..<epochs) {
            if (debug) debugTools.run { archiveWeights(); archiveDelta(); archiveBias() }
            val accuracy = DoubleArray(data.size())
            data.shuffle()
            var totalLoss = 0.0
            for (batchIndex in 0 until batchCount) {
                val lossDerivationSum = DoubleArray(layers.last().nbNeurons)
                val futures = mutableListOf<CompletableFuture<*>>()
                for (index in 0 until batchSize) {
                    futures.add(CompletableFuture.runAsync({
                        val sample = data.get(batchIndex * batchSize + index)
                        val (inputs, target) = sample
                        val outputs = predict(inputs)

                        for (i in outputs.indices) {
                            lossDerivationSum[i] += this.loss.derivative(outputs[i], target[i])
                        }
                        accuracy[batchIndex * batchSize + index] = getAccuracy(outputs, target)
                        totalLoss += this.loss.averageLoss(outputs, target)
                    }, threadPool))
                }
                awaitFutures(futures)
                this.optimizer.optimize(this, lossDerivationSum.map { it / batchSize }.toDoubleArray())
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

        if (debug) debugTools.run { printDeltas(); printWeights();printBias() }
        threadPool.shutdown()
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
            for (i in 0..<nbNeurons) {
                line = bufferedReader.readLine()
                if (line.isEmpty()) continue
                val bias = line.split(";").first().toDouble()
                val weigthsString = line.split(";").last()
                if (weigthsString.isEmpty()) continue
                layer.neurons[i].weights = weigthsString.split(" ").map { it.toDouble() }.toDoubleArray()
                layer.neurons[i].bias = bias
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
            for (neuron in layer.neurons) {
                val str = "${neuron.bias};${neuron.weights.joinToString(" ")}"
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