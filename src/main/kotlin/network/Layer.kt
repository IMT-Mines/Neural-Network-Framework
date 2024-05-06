package main.kotlin.network

class Layer(
    nbNeurons: Int,
    var activationFunction: ActivationFunction = Linear,
    private var useBias: Boolean = true
) {

    var neurons: Array<Neuron> = Array(nbNeurons) { Neuron(0.0) }
    private var bias: Double = 0.0

    fun initialize(nbInputs: Int = 0) {
        bias = if (useBias) Math.random() * 2 - 1 else 0.0
        for (i in neurons.indices) {
            val neuron = Neuron(bias)
            neuron.initialize(nbInputs)
            neurons[i] = neuron
        }
    }

    fun compute(inputs: DoubleArray): DoubleArray {
        var outputs = DoubleArray(neurons.size)
        for (i in neurons.indices) {
            outputs[i] = neurons[i].compute(inputs)
        }
        outputs = activationFunction.activate(outputs)
        for (i in neurons.indices) {
            neurons[i].output = outputs[i]
        }
        return outputs
    }

    fun getDerivativeOfEachNeuron(): DoubleArray {
        val derivatives = DoubleArray(neurons.size)
        for (i in neurons.indices) {
            derivatives[i] = neurons[i].output
        }
        return activationFunction.derivative(derivatives)
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