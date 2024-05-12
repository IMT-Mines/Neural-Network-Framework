package main.kotlin.network

import kotlin.math.exp
import kotlin.math.tanh


interface Activation {
    fun activate(z: DoubleArray): DoubleArray
    fun derivative(z: DoubleArray): DoubleArray
}

object Sigmoid : Activation {
    override fun derivative(z: DoubleArray): DoubleArray {
        return z.map { it * (1 - it) }.toDoubleArray()
    }

    override fun activate(z: DoubleArray): DoubleArray {
        return z.map { 1 / (1 + exp(-it)) }.toDoubleArray()
    }
}

object ReLU : Activation {
    override fun derivative(z: DoubleArray): DoubleArray {
        return z.map { if (it > 0) 1.0 else 0.0 }.toDoubleArray()
    }

    override fun activate(z: DoubleArray): DoubleArray {
        return z.map { if (it > 0) it else 0.0 }.toDoubleArray()
    }
}

object LeakyReLU : Activation {
    override fun derivative(z: DoubleArray): DoubleArray {
        return z.map { if (it > 0) 1.0 else 0.01 }.toDoubleArray()
    }

    override fun activate(z: DoubleArray): DoubleArray {
        return z.map { if (it > 0) it else 0.01 * it }.toDoubleArray()
    }
}

object Tanh : Activation {
    override fun derivative(z: DoubleArray): DoubleArray {
        return z.map { 1 - it * it }.toDoubleArray()
    }

    override fun activate(z: DoubleArray): DoubleArray {
        return z.map { tanh(it) }.toDoubleArray()
    }
}

object Linear : Activation {
    override fun derivative(z: DoubleArray): DoubleArray {
        return z
    }

    override fun activate(z: DoubleArray): DoubleArray {
        return z
    }
}

object Softmax : Activation {

    override fun activate(z: DoubleArray): DoubleArray {
        val max = z.maxOrNull() ?: 0.0
        val expSum = z.sumOf { exp(it - max) }
        return z.map { exp(it - max) / expSum }.toDoubleArray()
    }

    override fun derivative(z: DoubleArray): DoubleArray {
        return z.map { it * (1 - it) }.toDoubleArray()
    }
}