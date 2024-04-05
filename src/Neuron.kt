class Neuron(private val activationFunction: ActivationFunction) {

    private lateinit var weights: DoubleArray
    private var bias: Double = 0.0

    fun initialize(nbInputs: Int) {
        weights = DoubleArray(nbInputs) { Math.random() * 2 - 1 }
        bias = Math.random() * 2 - 1
    }

    fun compute(inputs: DoubleArray): Double {
        require(inputs.size == weights.size) { "The number of inputs must be equal to the number of weights" }
        var output = bias
        for (i in inputs.indices) {
            output += inputs[i] * weights[i]
        }
        return activationFunction.activate(output)
    }

    override fun toString(): String {
        return "Neuron(weights=$weights, bias=$bias)"
    }
}