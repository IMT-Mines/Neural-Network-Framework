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
        return activate(x) * (1 - activate(x))
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

object Tanh : ActivationFunction {
    override fun activate(x: Double): Double {
        return kotlin.math.tanh(x)
    }

    override fun derivative(x: Double): Double {
        return 1 - activate(x) * activate(x)
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