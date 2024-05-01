package main.kotlin.network

import kotlin.math.exp


interface ActivationFunction {
    fun activate(x: Double): Double
    fun derivative(x: Double): Double
    fun activate(z: DoubleArray): DoubleArray {
        throw NotImplementedError("Activation not implemented for this activation function")
    }
    fun derivative(z: DoubleArray): DoubleArray {
        throw NotImplementedError("Derivative not implemented for this activation function")
    }
}

object Sigmoid : ActivationFunction {
    override fun activate(x: Double): Double {
        return 1 / (1 + exp(-x))
    }

    override fun derivative(x: Double): Double {
        return x * (1 - x)
    }
}

object ReLU : ActivationFunction {
    override fun activate(x: Double): Double {
        return if (x > 0) x else 0.0
    }

    override fun derivative(x: Double): Double {
        return if (x > 0) 1.0 else 0.0
    }
}

object LeakyReLU : ActivationFunction {
    override fun activate(x: Double): Double {
        return if (x > 0) x else 0.01 * x
    }

    override fun derivative(x: Double): Double {
        return if (x > 0) 1.0 else 0.01
    }
}

object Tanh : ActivationFunction {
    override fun activate(x: Double): Double {
        return kotlin.math.tanh(x)
    }

    override fun derivative(x: Double): Double {
        return 1 - x * x
    }
}

object Linear : ActivationFunction {
    override fun activate(x: Double): Double {
        return x
    }

    override fun derivative(x: Double): Double {
        return 1.0
    }
}

object Softmax : ActivationFunction {
    override fun activate(x: Double): Double {
        return x
    }

    override fun derivative(x: Double): Double {
        return x
    }

    override fun activate(z: DoubleArray): DoubleArray {
        val expZ = z.map { exp(it) }
        val sumExpZ = expZ.sum()
        return expZ.map { it / sumExpZ }.toDoubleArray()
    }

    override fun derivative(z: DoubleArray): DoubleArray {
        val n = z.size
        val jacobian = Array(n) { DoubleArray(n) }

        val softmaxZ = activate(z)

        for (i in 0 until n) {
            for (j in 0 until n) {
                jacobian[i][j] = if (i == j) softmaxZ[i] * (1 - softmaxZ[j]) else -softmaxZ[i] * softmaxZ[j]
            }
        }

        // Calculer les dérivées par rapport à chaque sortie
        val derivatives = DoubleArray(n)
        for (i in 0 until n) {
            var derivativeSum = 0.0
            for (j in 0 until n) {
                derivativeSum += jacobian[j][i]
            }
            derivatives[i] = derivativeSum
        }

        return derivatives
    }
}