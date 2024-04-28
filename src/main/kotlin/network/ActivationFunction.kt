package main.kotlin.network

import kotlin.math.exp


interface ActivationFunction {
    fun activate(x: Double): Double
    fun derivative(x: Double): Double
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

//object Softmax : ActivationFunction() {
//
//    override fun activate(x: Double): Double {
//        return x
//    }
//
//    override fun activate(z: DoubleArray): DoubleArray {
//        val maxZ = z.maxOrNull() ?: throw IllegalArgumentException("Input array must not be empty")
//        val expZ = z.map { exp(it - maxZ) }
//        val sumExpZ = expZ.sum()
//        return expZ.map { it / sumExpZ }.toDoubleArray()
//    }
//
//    override fun derivative(z: DoubleArray): DoubleArray {
//        val softmaxZ = activate(z)
//        val n = softmaxZ.size
//        val jacobian = DoubleArray(n) { i ->
//            softmaxZ[i] * (1 - softmaxZ[i])
//        }
//        return jacobian
//    }
//}