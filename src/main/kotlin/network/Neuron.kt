package main.kotlin.network

class Neuron {
    var weights: DoubleArray = doubleArrayOf()
    var output: Double = 0.0
    var delta: Double = 0.0

    fun initialize(nbInputs: Int = 0) {
        weights = DoubleArray(nbInputs) { Math.random() * 2 - 1 }
    }

    fun compute(inputs: DoubleArray): Double {
        var output = 0.0
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
        }, value=$output"
    }
}