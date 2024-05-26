package main.kotlin.network

import main.kotlin.utils.Utils.Companion.awaitFutures
import java.util.concurrent.CompletableFuture
import java.util.concurrent.Executors
import kotlin.math.sqrt

abstract class Optimizer {

    private val threadsThreshold: Int = 2;

    private val threadPool = Executors.newFixedThreadPool(512)

    class Gradients(val weightGradients: Array<Array<DoubleArray>>, val biasGradients: Array<DoubleArray>)

    fun minimize(model: NeuralNetwork, targets: List<DoubleArray>, outputs: List<DoubleArray>) {
        val layers = model.layers
        val weightGradients = Array(layers.size) { layerIndex ->
            Array(layers[layerIndex].neurons.size) { neuronIndex ->
                DoubleArray(layers[layerIndex].neurons[neuronIndex].weights.size) { 0.0 }
            }
        }
        val biasGradients = Array(layers.size) { layerIndex ->
            DoubleArray(layers[layerIndex].neurons.size) { 0.0 }
        }
        val gradients = Gradients(weightGradients, biasGradients)

        calcualteGradiants(model, gradients, targets, outputs)

        update(model, gradients, targets.size)
    }

    private fun calcualteGradiants(
        model: NeuralNetwork,
        gradients: Gradients,
        targets: List<DoubleArray>,
        outputs: List<DoubleArray>
    ) {
        if (targets.size < threadsThreshold) {
            for (i in targets.indices) {
                calculateGradiant(model, gradients, targets[i], outputs[i])
            }
        } else {
            val futures = mutableListOf<CompletableFuture<*>>()
            for (i in targets.indices) {
                futures.add(CompletableFuture.runAsync({
                    calculateGradiant(model, gradients, targets[i], outputs[i])
                }, threadPool))
            }
            awaitFutures(futures)
        }
    }

    abstract fun update(model: NeuralNetwork, gradients: Gradients, batchSize: Int)

    abstract fun calculateGradiant(
        model: NeuralNetwork,
        gradients: Gradients,
        targets: DoubleArray,
        outputs: DoubleArray
    )

}

class SGD(private var learningRate: Double = 0.01) : Optimizer() {

    override fun update(model: NeuralNetwork, gradients: Gradients, batchSize: Int) {
        for (layerIndex in 1 until model.layers.size) {
            for (neuronIndex in model.layers[layerIndex].neurons.indices) {
                val neuron = model.layers[layerIndex].neurons[neuronIndex]
                for (weightIndex in neuron.weights.indices) {
                    neuron.weights[weightIndex] -= learningRate * gradients.weightGradients[layerIndex][neuronIndex][weightIndex] / batchSize
                }
                neuron.bias -= learningRate * gradients.biasGradients[layerIndex][neuronIndex] / batchSize
            }
        }
    }

    override fun calculateGradiant(
        model: NeuralNetwork,
        gradients: Gradients,
        targets: DoubleArray,
        outputs: DoubleArray
    ) {
        val layers = model.layers
        val lastLayer = layers.last()

        val activationOutputsLayerDerivative = lastLayer.getActivationDerivative()
        for (neuronIndex in lastLayer.neurons.indices) {
            val neuron = lastLayer.neurons[neuronIndex]
            for (weightIndex in neuron.weights.indices) {
                val derivativeLoss = model.loss.derivative(outputs[neuronIndex], targets[neuronIndex])
                neuron.delta = derivativeLoss * activationOutputsLayerDerivative[neuronIndex]
                val previousLayerNeuron = layers[layers.size - 2].neurons[weightIndex]
                gradients.weightGradients[layers.size - 1][neuronIndex][weightIndex] += neuron.delta * previousLayerNeuron.output
            }
            gradients.biasGradients[layers.size - 1][neuronIndex] += neuron.delta
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
                    gradients.weightGradients[layerIndex][neuronIndex][weightIndex] += neuron.delta * previousLayerNeuron.output
                }
                gradients.biasGradients[layerIndex][neuronIndex] += neuron.delta
            }
        }
    }
}

class Adam(
    private var learningRate: Double = 0.01,
    private var beta1: Double = 0.9,
    private var beta2: Double = 0.999,
    private var epsilon: Double = 1e-8
) : Optimizer() {

      override fun update(model: NeuralNetwork, gradients: Gradients, batchSize: Int) {
        for (layerIndex in 1 until model.layers.size) {
            for (neuronIndex in model.layers[layerIndex].neurons.indices) {
                val neuron = model.layers[layerIndex].neurons[neuronIndex]
                for (weightIndex in neuron.weights.indices) {
                    neuron.weights[weightIndex] -= learningRate * gradients.weightGradients[layerIndex][neuronIndex][weightIndex] / batchSize
                }
                neuron.bias -= learningRate * gradients.biasGradients[layerIndex][neuronIndex] / batchSize
            }
        }
    }

    override fun calculateGradiant(
        model: NeuralNetwork,
        gradients: Gradients,
        targets: DoubleArray,
        outputs: DoubleArray
    ) {
        val layers = model.layers
        val lastLayer = layers.last()

        val activationOutputsLayerDerivative = lastLayer.getActivationDerivative()
        for (neuronIndex in lastLayer.neurons.indices) {
            val neuron = lastLayer.neurons[neuronIndex]
            for (weightIndex in neuron.weights.indices) {
                val derivativeLoss = model.loss.derivative(outputs[neuronIndex], targets[neuronIndex])
                neuron.delta = derivativeLoss * activationOutputsLayerDerivative[neuronIndex]
                val previousLayerNeuron = layers[layers.size - 2].neurons[weightIndex]
                val gradient = neuron.delta * previousLayerNeuron.output
                neuron.m[weightIndex] = beta1 * neuron.m[weightIndex] + (1 - beta1) * gradient
                neuron.v[weightIndex] = beta2 * neuron.v[weightIndex] + (1 - beta2) * gradient * gradient
                val mHat = neuron.m[weightIndex] / (1 - beta1)
                val vHat = neuron.v[weightIndex] / (1 - beta2)
                gradients.weightGradients[layers.size - 1][neuronIndex][weightIndex] += mHat / (sqrt(vHat) + epsilon)
            }
            gradients.biasGradients[layers.size - 1][neuronIndex] += neuron.delta
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
                    gradients.weightGradients[layerIndex][neuronIndex][weightIndex] += mHat / (sqrt(vHat) + epsilon)
                }
                gradients.biasGradients[layerIndex][neuronIndex] += neuron.delta
            }
        }
    }
}