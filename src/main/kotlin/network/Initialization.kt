package main.kotlin.network

import main.kotlin.utils.Utils
import kotlin.math.sqrt

interface Initialization {

    fun initialize(fanIn: Int, fanOut: Int): DoubleArray

}

object RandomInitialization : Initialization {

    override fun initialize(fanIn: Int, fanOut: Int): DoubleArray {
        val weights = DoubleArray(fanIn)
        for (i in 0..<fanIn) {
            weights[i] = Math.random() * 2 - 1
        }
        return weights
    }
}

object UniformDistributionInitialization : Initialization {

    override fun initialize(fanIn: Int, fanOut: Int): DoubleArray {
        val weights = DoubleArray(fanIn)
        val range = 1 / sqrt(fanIn.toDouble())
        for (i in 0..<fanIn) {
            weights[i] = (Math.random() * 2 - 1) * range
        }
        return weights
    }
}

object UniformXavierGlorotInitialization : Initialization {

    override fun initialize(fanIn: Int, fanOut: Int): DoubleArray {
        val weights = DoubleArray(fanIn)
        val range = sqrt(6.0 / (fanIn + fanOut + 1))
        for (i in 0..<fanIn) {
            weights[i] = (Math.random() * 2 - 1) * range
        }
        return weights
    }
}

/**
 * This initialization works well with Linear, Tanh, Softmax and Sigmoid activation functions
 */
object NormalXavierGlorotInitialization : Initialization {

    override fun initialize(fanIn: Int, fanOut: Int): DoubleArray {
        val weights = DoubleArray(fanIn)
        val fanAvg = (fanIn + fanOut) / 2.0;
        for (i in 0..<fanIn) {
            weights[i] = Utils.randomNormal(0.0, 1 / fanAvg)
        }
        return weights
    }
}

/**
 * This initialization works well with ReLU activation function
 */
object NormalHeInitialization : Initialization {

    override fun initialize(fanIn: Int, fanOut: Int): DoubleArray {
        val weights = DoubleArray(fanIn)
        for (i in 0..<fanIn) {
            weights[i] = Utils.randomNormal(0.0, 2 / fanIn.toDouble())
        }
        return weights
    }
}