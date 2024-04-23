class Neuron(val activationFunction: ActivationFunction, private val useBias: Boolean = true) {

    var weights: DoubleArray = doubleArrayOf()
    var value: Double = 0.0
    var backPropagation: Double = 0.0
    private var bias: Double = 0.0

    fun initialize(nbInputs: Int = 0, values: DoubleArray = doubleArrayOf()) {
        if (values.isNotEmpty()) {
            weights = values
            return
        }
        weights = DoubleArray(nbInputs) { Math.random() * 2 - 1 }
        bias = if (useBias) Math.random() * 2 - 1 else 0.0
    }

    fun compute(inputs: DoubleArray): Double {
        //require(inputs.size == weights.size) { "The number of inputs must be equal to the number of weights" }
        var output = bias
        for (i in inputs.indices) {
            output += inputs[i] * weights[i]
        }
        this.value = activationFunction.activate(output)
        return this.value
    }

    override fun toString(): String {
        return "weights=${
            weights.joinToString {
                it.toString()
            }
        }, bias=$bias, value=$value"
    }
}