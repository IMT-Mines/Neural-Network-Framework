package main.kotlin.network

class Neuron(private val useBias: Boolean = true) {
    var weights: DoubleArray = doubleArrayOf()
    var output: Double = 0.0
    var delta: Double = 0.0
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
        var output = bias
        for (i in inputs.indices) {
            output += inputs[i] * weights[i]
        }
        this.output = output
        return this.output
    }

    override fun toString(): String {
        return "weights=${
            weights.joinToString {
                it.toString()
            }
        }, bias=$bias, value=$output"
    }
}