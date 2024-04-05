class NeuralNetwork(private val layers: List<Layer>) {

    fun predict(inputs: DoubleArray): DoubleArray {
        var outputs = inputs
        for (layer in layers) {
            outputs = layer.compute(outputs)
        }
        return outputs
    }

    override fun toString(): String {
        val sb = StringBuilder()
        for (layer in layers) {
            sb.append(layer.toString())
            sb.append("\n")
        }
        return sb.toString()
    }
}