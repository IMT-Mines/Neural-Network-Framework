package main.kotlin.network

import main.kotlin.reinforcement.Environment
import main.kotlin.reinforcement.Experience
import main.kotlin.reinforcement.ReplayMemory
import main.kotlin.train.Data
import main.kotlin.utils.Chart
import main.kotlin.utils.DebugTools
import org.jetbrains.kotlinx.kandy.util.color.Color

interface Train {
    fun fit(model: NeuralNetwork, epochs: Int, debug: Boolean = false)
    fun test(model: NeuralNetwork)
}

class QLearning<E : Enum<E>>(
    private val environment: Environment<E>,
    private val maxIteration: Int,
    private val batchSize: Int = 64,
    private val gamma: Double = 0.9,
    maxMemory: Int = 10000
) : Train {

    private val memory = ReplayMemory(maxMemory)

    override fun fit(model: NeuralNetwork, epochs: Int, debug: Boolean) {
        val lossChart: MutableMap<Int, Double> = mutableMapOf()
        println("\n======================= TRAINING =======================\n")
        for (epoch in 0 until epochs) {
            var state = environment.reset()
            var totalLoss = 0.0
            var nbIteration = 0

            for (iteration in 0 until maxIteration) {
                val qValues = model.predict(state)
                val action = qValues.withIndex().maxByOrNull { it.value }?.index!!

                val (nextState, reward) = environment.step(action)

                memory.add(Experience(state, action, nextState, reward))

                state = nextState

                if (memory.size() >= batchSize) {
                    val batch = memory.sample(batchSize)
                    val targets = mutableListOf<DoubleArray>()
                    val outputs = mutableListOf<DoubleArray>()

                    for (experience in batch) {
                        val nextQValues = model.predict(experience.nextState)
                        val nextAction = nextQValues.withIndex().maxByOrNull { it.value }?.index!!

                        val target = experience.reward + gamma * nextQValues[nextAction]

                        // Update the target for the action taken only
                        val output = model.predict(experience.state).copyOf()
                        output[experience.action] = target

                        targets.add(output)
                        outputs.add(model.predict(experience.state))
                    }

                    model.optimizer.minimize(model, targets, outputs)

                    for (i in targets.indices) {
                        totalLoss += model.loss.averageLoss(outputs[i], targets[i])
                    }
                    nbIteration += batchSize
                }

                if (environment.isDone()) {
                    println("Finished in $iteration iterations")
                    break
                }
            }
            lossChart[epoch] = totalLoss / nbIteration
            println("Epoch: %d | Training Loss: %10.4f".format(epoch, totalLoss / nbIteration))
        }
        Chart.lineChart(lossChart, "Model loss", "Epoch", "Loss", Color.BLUE, "src/main/resources/plots")
    }


    override fun test(model: NeuralNetwork) {
        for (i in 0 until 10) {
            var state = environment.reset()
            for (j in 0 until 1000) {
                val action = model.predict(state).withIndex().maxByOrNull { it.value }?.index
                val (nextState, reward) = environment.step(action!!)
                println("Action: $action | Reward: $reward")
                println(environment)
                state = nextState
                if (environment.isDone()) {
                    println("Done")
                    break
                }
            }
        }
    }


}

class StandardTraining(
    private var datas: Pair<Data, Data>, private var batchSize: Int = 1
) : Train {

    override fun fit(model: NeuralNetwork, epochs: Int, debug: Boolean) {
        val data = datas.first
        if (data.size() % batchSize != 0) throw IllegalArgumentException("The batch size must be a multiple of the data size")

        val debugTools = DebugTools(model)
        println("\n======================= TRAINING =======================\n")
        val lossChart: MutableMap<Int, Double> = mutableMapOf()
        val accuracyChart: MutableMap<Int, Double> = mutableMapOf()

        val batchCount = data.size() / batchSize

        for (epoch in 0..<epochs) {
            if (debug) debugTools.run { archiveWeights(); archiveDelta(); archiveBias() }
            val accuracy = DoubleArray(data.size())
            data.shuffle()
            var totalLoss = 0.0
            for (batchIndex in 0 until batchCount) {
                val targetsList = mutableListOf<DoubleArray>()
                val outputsList = mutableListOf<DoubleArray>()
                for (index in 0 until batchSize) {
                    val sample = data.get(batchIndex * batchSize + index)
                    val (inputs, target) = sample
                    val outputs = model.predict(inputs)
                    accuracy[batchIndex * batchSize + index] = getAccuracy(outputs, target)
                    totalLoss += model.loss.averageLoss(outputs, target)
                    targetsList.add(target)
                    outputsList.add(outputs)
                }
                model.optimizer.minimize(model, targetsList, outputsList)
            }

            accuracyChart[epoch] = accuracy.average()
            lossChart[epoch] = totalLoss / data.size()
            println(
                "Epoch: %d | Training Loss: %10.4f | Accuracy: %10.2f".format(
                    epoch,
                    totalLoss / data.size(),
                    accuracy.average()
                )
            )
        }

        if (debug) debugTools.run { printDeltas(); printWeights();printBias() }
        Chart.lineChart(accuracyChart, "Model accuracy", "Epoch", "Accuracy", Color.GREEN, "src/main/resources/plots")
        Chart.lineChart(lossChart, "Model loss", "Epoch", "Loss", Color.BLUE, "src/main/resources/plots")
    }

    override fun test(model: NeuralNetwork) {
        val data = datas.second
        println("\n======================= TESTING =======================\n")
        var accuracy = 0.0
        for (index in 0..<data.size()) {
            val sample = data.get(index)
            val inputs = sample.first
            val target = sample.second
            val outputs = model.predict(inputs)

            accuracy += getAccuracy(outputs, target)
            println(
                "Output: ${outputs.joinToString { "%.2f".format(it) }} | Target: ${
                    target.joinToString {
                        "%.2f".format(
                            it
                        )
                    }
                }"
            )
        }
        println("\nThe Accuracy on the test set is: ${accuracy / data.size()}")
    }

    private fun getAccuracy(outputs: DoubleArray, target: DoubleArray): Double {
        return if (outputs.size == 1) {
            when {
                outputs[0] >= 0.5 && target[0] == 1.0 -> 1.0
                outputs[0] < 0.5 && target[0] == 0.0 -> 1.0
                else -> 0.0
            }
        } else {
            if (outputs.withIndex().maxByOrNull { it.value }?.index == target.withIndex()
                    .maxByOrNull { it.value }?.index
            ) 1.0 else 0.0
        }
    }
}

class PPOTraining<E : Enum<E>>(
    private val environment: Environment<E>,
    private val gamma: Double = 0.99,
    private val epsilon: Double = 0.2
) : Train {

    override fun fit(model: NeuralNetwork, epochs: Int, debug: Boolean) {
        for (epoch in 0 until epochs) {
            val states = mutableListOf<DoubleArray>()
            val actions = mutableListOf<E>()
            val rewards = mutableListOf<Double>()
            val logProbs = mutableListOf<Double>()

            val maxSteps = 1000
            var state = environment.reset()
            for (step in 0 until maxSteps) {
//                val (action, logProb) = selectAction(state, model)
//                val (nextState, reward) = environment.step(action)
//                states.add(state)
//                actions.add(action)
//                rewards.add(reward)
//                logProbs.add(logProb)

//                state = nextState
                if (environment.isDone()) {
                    break
                }
            }
//            val returns = calculateReturns(rewards, gamma)
//            optimizeModel(states, actions, logProbs, returns, model, epsilon)
        }
    }

    override fun test(model: NeuralNetwork) {
        // Not implemented
    }
}