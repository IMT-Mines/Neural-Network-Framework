package main.kotlin.network

class Layer(
    nbNeurons: Int,
    var activationFunction: ActivationFunction = Linear,
    private var useBias: Boolean = false
) {

    var neurons: Array<Neuron> = Array(nbNeurons) { Neuron(activationFunction) }

    fun initialize(nbInputs: Int = 0) {
        for (i in neurons.indices) {
            val neuron = Neuron(activationFunction, useBias)
            neuron.initialize(nbInputs)
            neurons[i] = neuron
        }
    }

    fun compute(inputs: DoubleArray): DoubleArray {
        val outputs = DoubleArray(neurons.size)
        for (i in neurons.indices) {
            outputs[i] = neurons[i].compute(inputs)
        }
        return outputs
    }

    override fun toString(): String {
        val sb = StringBuilder()
        for (neuron in neurons) {
            sb.append(neuron.toString())
            sb.append("\n")
        }
        return sb.toString()
    }
}