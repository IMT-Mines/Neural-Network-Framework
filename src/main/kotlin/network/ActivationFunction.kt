package main.kotlin.network

import kotlin.math.exp


interface ActivationFunction {
    fun activate(z: DoubleArray): DoubleArray
    fun derivative(z: DoubleArray): DoubleArray
}

object Sigmoid : ActivationFunction {
    private fun activate(x: Double): Double {
        return 1 / (1 + exp(-x))
    }

    private fun derivative(x: Double): Double {
        return x * (1 - x)
    }

    override fun derivative(z: DoubleArray): DoubleArray {
        return z.map { derivative(it) }.toDoubleArray()
    }

    override fun activate(z: DoubleArray): DoubleArray {
        return z.map { activate(it) }.toDoubleArray()
    }
}

object ReLU : ActivationFunction {
    private fun activate(x: Double): Double {
        return if (x > 0) x else 0.0
    }

    private fun derivative(x: Double): Double {
        return if (x > 0) 1.0 else 0.0
    }

    override fun derivative(z: DoubleArray): DoubleArray {
        return z.map { derivative(it) }.toDoubleArray()
    }

    override fun activate(z: DoubleArray): DoubleArray {
        return z.map { activate(it) }.toDoubleArray()
    }
}

object LeakyReLU : ActivationFunction {
    private fun activate(x: Double): Double {
        return if (x > 0) x else 0.01 * x
    }

    private fun derivative(x: Double): Double {
        return if (x > 0) 1.0 else 0.01
    }

    override fun derivative(z: DoubleArray): DoubleArray {
        return z.map { derivative(it) }.toDoubleArray()
    }

    override fun activate(z: DoubleArray): DoubleArray {
        return z.map { activate(it) }.toDoubleArray()
    }
}

object Tanh : ActivationFunction {
    private fun activate(x: Double): Double {
        return kotlin.math.tanh(x)
    }

    private fun derivative(x: Double): Double {
        return 1 - x * x
    }

    override fun derivative(z: DoubleArray): DoubleArray {
        return z.map { derivative(it) }.toDoubleArray()
    }


    override fun activate(z: DoubleArray): DoubleArray {
        return z.map { activate(it) }.toDoubleArray()
    }
}

object Linear : ActivationFunction {
    private fun activate(x: Double): Double {
        return x
    }

    private fun derivative(x: Double): Double {
        return 1.0
    }

    override fun derivative(z: DoubleArray): DoubleArray {
        return z.map { derivative(it) }.toDoubleArray()
    }

    override fun activate(z: DoubleArray): DoubleArray {
        return z.map { activate(it) }.toDoubleArray()
    }
}

object Softmax : ActivationFunction {

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