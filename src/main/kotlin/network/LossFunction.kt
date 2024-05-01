package main.kotlin.network

import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow

interface LossFunction {
    fun lossCalcul(output: Double, target: Double): Double
    fun derivative(output: Double, target: Double): Double
    fun loss(outputs: DoubleArray, targets: DoubleArray): Double
}

object SquaredError : LossFunction {
    override fun lossCalcul(output: Double, target: Double): Double {
        return (1.0 / 2.0) * (output - target).pow(2.0)
    }

    override fun derivative(output: Double, target: Double): Double {
        return output - target
    }

    override fun loss(outputs: DoubleArray, targets: DoubleArray): Double {
        var totalLoss = 0.0
        for (i in outputs.indices) {
            totalLoss += lossCalcul(outputs[i], targets[i])
        }
        return totalLoss
    }
}

object MeanSquaredError : LossFunction {
    override fun lossCalcul(output: Double, target: Double): Double {
        return (output - target).pow(2.0)
    }

    override fun derivative(output: Double, target: Double): Double {
        return output - target
    }

    override fun loss(outputs: DoubleArray, targets: DoubleArray): Double {
        var totalLoss = 0.0
        for (i in outputs.indices) {
            totalLoss += lossCalcul(outputs[i], targets[i])
        }
        return 1.0 / outputs.size * totalLoss
    }
}

object BinaryCrossEntropy : LossFunction {
    private const val EPSILON = 1e-15

    override fun lossCalcul(output: Double, target: Double): Double {
        val clippedOutput = max(min(output, 1.0 - EPSILON), EPSILON)
        return -target * ln(clippedOutput) - (1 - target) * ln(1 - clippedOutput)
    }

    override fun derivative(output: Double, target: Double): Double {
        val clippedOutput = max(min(output, 1.0 - EPSILON), EPSILON)
        return (clippedOutput - target) / (clippedOutput * (1 - clippedOutput))
    }

    override fun loss(outputs: DoubleArray, targets: DoubleArray): Double {
        var totalLoss = 0.0
        for (i in outputs.indices) {
            totalLoss += lossCalcul(outputs[i], targets[i])
        }
        return 1.0 / outputs.size * totalLoss
    }
}

object CategoricalCrossEntropy : LossFunction {
    private const val EPSILON = 1e-15

    override fun lossCalcul(output: Double, target: Double): Double {
        val clippedOutput = max(min(output, 1.0 - EPSILON), EPSILON)
        return -target * ln(clippedOutput)
    }

    override fun derivative(output: Double, target: Double): Double {
        return (output - target)
    }

    override fun loss(outputs: DoubleArray, targets: DoubleArray): Double {
        var totalLoss = 0.0
        for (i in outputs.indices) {
            totalLoss += lossCalcul(outputs[i], targets[i])
        }
        return totalLoss
    }
}