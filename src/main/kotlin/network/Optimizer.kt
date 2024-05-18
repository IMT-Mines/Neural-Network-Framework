package main.kotlin.network

import kotlin.math.sqrt

interface Optimizer {

    fun optimize(model: NeuralNetwork, loss: DoubleArray)

}

class SGD(private var learningRate: Double = 0.01) : Optimizer {

    override fun optimize(model: NeuralNetwork, loss: DoubleArray) {
        val layers = model.layers
        val lastLayer = layers.last()

        val activationOutputsLayerDerivative = lastLayer.getActivationDerivative()
        for (neuronIndex in lastLayer.neurons.indices) {
            val neuron = lastLayer.neurons[neuronIndex]
            for (weightIndex in neuron.weights.indices) {
                neuron.delta = loss[neuronIndex] * activationOutputsLayerDerivative[neuronIndex]
                val previousLayerNeuron = layers[layers.size - 2].neurons[weightIndex]
                neuron.weights[weightIndex] -= learningRate * neuron.delta * previousLayerNeuron.output
            }
            neuron.bias -= learningRate * neuron.delta
        }

        for (layerIndex in layers.size - 2 downTo 1) {
            val activationOutputsDerivatives = layers[layerIndex].getActivationDerivative()
            for (neuronIndex in layers[layerIndex].neurons.indices) {
                val neuron = layers[layerIndex].neurons[neuronIndex]
                for (weightIndex in neuron.weights.indices) {
                    var lossGradiant = 0.0
                    for (nextLayerNeuron in layers[layerIndex + 1].neurons.indices) {
                        val beforeNeuron = layers[layerIndex + 1].neurons[nextLayerNeuron]
                        lossGradiant += beforeNeuron.weights[neuronIndex] * beforeNeuron.delta
                    }
                    neuron.delta = lossGradiant * activationOutputsDerivatives[neuronIndex]
                    val previousLayerNeuron = layers[layerIndex - 1].neurons[weightIndex]
                    neuron.weights[weightIndex] -= learningRate * neuron.delta * previousLayerNeuron.output
                }
                neuron.bias -= learningRate * neuron.delta
            }
        }
//        val futures = mutableListOf<CompletableFuture<*>>()
//        futures.add(CompletableFuture.runAsync({
//        }, threadPool))
//        awaitFutures(futures)
    }
}

class Adam(
    private var learningRate: Double = 0.01,
    private var beta1: Double = 0.9,
    private var beta2: Double = 0.999,
    private var epsilon: Double = 1e-8
) : Optimizer {

    override fun optimize(model: NeuralNetwork, loss: DoubleArray) {
        val layers = model.layers
        val lastLayer = layers.last()

        val activationOutputsLayerDerivative = lastLayer.getActivationDerivative()
        for (neuronIndex in lastLayer.neurons.indices) {
            val neuron = lastLayer.neurons[neuronIndex]
            for (weightIndex in neuron.weights.indices) {
                neuron.delta = loss[neuronIndex] * activationOutputsLayerDerivative[neuronIndex]
                val previousLayerNeuron = layers[layers.size - 2].neurons[weightIndex]
                val gradient = neuron.delta * previousLayerNeuron.output
                neuron.m[weightIndex] = beta1 * neuron.m[weightIndex] + (1 - beta1) * gradient
                neuron.v[weightIndex] = beta2 * neuron.v[weightIndex] + (1 - beta2) * gradient * gradient
                val mHat = neuron.m[weightIndex] / (1 - beta1)
                val vHat = neuron.v[weightIndex] / (1 - beta2)
                neuron.weights[weightIndex] -= learningRate * mHat / (sqrt(vHat) + epsilon)
                neuron.bias -= learningRate * neuron.delta
            }
        }

        for (layerIndex in layers.size - 2 downTo 1) {
            val activationOutputsDerivatives = layers[layerIndex].getActivationDerivative()
            for (neuronIndex in layers[layerIndex].neurons.indices) {
                val neuron = layers[layerIndex].neurons[neuronIndex]
                for (weightIndex in neuron.weights.indices) {
                    var lossGradiant = 0.0
                    for (nextLayerNeuron in layers[layerIndex + 1].neurons.indices) {
                        val beforeNeuron = layers[layerIndex + 1].neurons[nextLayerNeuron]
                        lossGradiant += beforeNeuron.weights[neuronIndex] * beforeNeuron.delta
                    }
                    neuron.delta = lossGradiant * activationOutputsDerivatives[neuronIndex]
                    val previousLayerNeuron = layers[layerIndex - 1].neurons[weightIndex]
                    val gradient = neuron.delta * previousLayerNeuron.output
                    neuron.m[weightIndex] = beta1 * neuron.m[weightIndex] + (1 - beta1) * gradient
                    neuron.v[weightIndex] = beta2 * neuron.v[weightIndex] + (1 - beta2) * gradient * gradient
                    val mHat = neuron.m[weightIndex] / (1 - beta1)
                    val vHat = neuron.v[weightIndex] / (1 - beta2)
                    neuron.weights[weightIndex] -= learningRate * mHat / (sqrt(vHat) + epsilon)
                }
                neuron.bias -= learningRate * neuron.delta
            }
        }


    }
}