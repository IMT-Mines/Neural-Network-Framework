import java.lang.Math.log
import kotlin.math.ln
import kotlin.math.pow

interface LossFunction {
    fun loss(output: Double, target: Double): Double
    fun derivative(output: Double, target: Double): Double
    fun totalLoss(outputs: DoubleArray, targets: DoubleArray): Double
}

object SquaredError : LossFunction {
    override fun loss(output: Double, target: Double): Double {
        return (1.0 / 2.0) * (output - target).pow(2.0)
    }

    override fun derivative(output: Double, target: Double): Double {
        return output - target
    }

    override fun totalLoss(outputs: DoubleArray, targets: DoubleArray): Double {
        var totalLoss = 0.0
        for (i in outputs.indices) {
            totalLoss += loss(outputs[i], targets[i])
        }
        return totalLoss
    }
}

object MeanSquaredError : LossFunction {
    override fun loss(output: Double, target: Double): Double {
        return (output - target).pow(2.0)
    }

    override fun derivative(output: Double, target: Double): Double {
        return output - target
    }

    override fun totalLoss(outputs: DoubleArray, targets: DoubleArray): Double {
        var totalLoss = 0.0
        for (i in outputs.indices) {
            totalLoss += loss(outputs[i], targets[i])
        }
        return 1.0 / outputs.size * totalLoss
    }
}

object BinaryCrossEntropy : LossFunction {
    override fun loss(output: Double, target: Double): Double {
        return -target * ln(output) - (1 - target) * ln(1 - output)
    }

    override fun derivative(output: Double, target: Double): Double {
        return (output - target) / (output * (1 - output))
    }

    override fun totalLoss(outputs: DoubleArray, targets: DoubleArray): Double {
        var totalLoss = 0.0
        for (i in outputs.indices) {
            totalLoss += loss(outputs[i], targets[i])
        }
        return 1.0 / outputs.size * totalLoss
    }



}

//TODO: j'ai commencé a implémenter la gestion dynamique des lossfunction, maintenant, il faut commencer a implementer
// Celles pour faire l'algorithme de trie, attention, a revoir pour les probleme RELU et TANH qui ont du mal
// A converger et à obtenir une erreur adaptée.

