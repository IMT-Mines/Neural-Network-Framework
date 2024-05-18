package main.kotlin.network

class Neuron(nbInputs: Int) {
    var weights: DoubleArray = DoubleArray(nbInputs)
    var bias: Double = Math.random() * 0.2
    var output: Double = 0.0
    var delta: Double = 0.0
    var m: DoubleArray = DoubleArray(nbInputs)
    var v: DoubleArray = DoubleArray(nbInputs)

    fun compute(inputs: DoubleArray): Double {
        var output = 0.0
        for (i in inputs.indices) {
            output += inputs[i] * weights[i]
        }

        this.output = output + bias
        return this.output
    }
}