package main.kotlin.network

import java.io.File

class NeuralNetwork(var trainingMethod: Train, var loss: Loss = SquaredError, var optimizer: Optimizer = SGD(0.001)) {

    val layers = mutableListOf<Layer>()

    fun fit(epoch: Int, debug: Boolean = false) {
        this.trainingMethod.fit(this, epoch, debug)
    }

    fun test() {
        this.trainingMethod.test(this)
    }

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

    fun copy(): NeuralNetwork {
        val model = NeuralNetwork(trainingMethod, loss, optimizer)
        for (layer in layers) {
            model.addLayer(Layer(layer.nbNeurons, layer.activation))
        }
        model.initialize()
        model.copyWeightsFrom(this)
        return model
    }

    fun copyWeightsFrom(source: NeuralNetwork) {
        for (i in this.layers.indices) {
            for (j in this.layers[i].neurons.indices) {
                this.layers[i].neurons[j].weights = source.layers[i].neurons[j].weights.copyOf()
                this.layers[i].neurons[j].bias = source.layers[i].neurons[j].bias
            }
        }
    }
}