package main.kotlin.network

class Layer(
    var nbNeurons: Int,
    var activation: Activation = Linear,
    var initialization: Initialization = RandomInitialization,
) {

    var neurons: Array<Neuron> = Array(nbNeurons) { Neuron(0) }

    fun buildLayer(nbInputs: Int = 0) {
        for (i in neurons.indices) {
            val neuron = Neuron(nbInputs)
            neurons[i] = neuron
        }
    }


    fun compute(inputs: DoubleArray): DoubleArray {
        var outputs = DoubleArray(nbNeurons)
        for (i in neurons.indices) {
            outputs[i] = neurons[i].compute(inputs)
        }
        outputs = activation.activate(outputs)
        for (i in neurons.indices) {
            neurons[i].output = outputs[i]
        }
        return outputs
    }

    fun getDerivativeOfEachNeuron(): DoubleArray {
        val derivatives = DoubleArray(nbNeurons)
        for (i in neurons.indices) {
            derivatives[i] = neurons[i].output
        }
        return activation.derivative(derivatives)
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