class Neuron(private val activationFunction: ActivationFunction, private val useBias: Boolean = true) {

    lateinit var weights: DoubleArray
    private var bias: Double = 0.0

    fun initialize(nbInputs: Int) {
        weights = DoubleArray(nbInputs) { Math.random() * 2 - 1 }
        bias = if (useBias) Math.random() * 2 - 1 else 0.0
    }

    fun compute(inputs: DoubleArray): Double {
        //require(inputs.size == weights.size) { "The number of inputs must be equal to the number of weights" }
        var output = bias
        for (i in inputs.indices) {
            output += inputs[i] * weights[i]
        }
        return activationFunction.activate(output)
    }

    override fun toString(): String {
        return "weights=${
            weights.joinToString {
                it.toString()
            }
        }, bias=$bias)"
    }
}