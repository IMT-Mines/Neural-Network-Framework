class NeuralNetwork(private var learningRate: Double) {

    private val layers = mutableListOf<Layer>()

    fun predict(inputs: DoubleArray): DoubleArray {
        var outputs = inputs
        for (layer in layers) {
            outputs = layer.compute(outputs)
        }
        return outputs
    }

    fun addLayer(layer: Layer) {
        layers.toMutableList().add(layer)
    }

    fun minimize(lossFunction: () -> Double) {
        for (layer in layers) {
            for (neuron in layer.neurons) {
                neuron.adamOptimizer(learningRate, 0.9, 0.999, 1e-8, lossFunction)
            }
        }
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