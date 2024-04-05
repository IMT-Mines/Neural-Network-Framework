class Layer(nbNeurons: Int, nbInputs: Int, activationFunction: ActivationFunction) {

    private val neurons: MutableList<Neuron> = mutableListOf()

    init {
        repeat(nbNeurons) {
            val neuron = Neuron(activationFunction)
            neuron.initialize(nbInputs)
            neurons.add(neuron)
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