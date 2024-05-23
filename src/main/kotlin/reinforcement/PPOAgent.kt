package main.kotlin.reinforcement

import main.kotlin.network.NeuralNetwork


class PPOAgent(private val model: NeuralNetwork, private val environment: Environment) {

    fun train(epochs: Int, batchSize: Int = 1, gamma: Double = 0.99, epsilon: Double = 0.2) {
        for (epoch in 0 until epochs) {
            val states = mutableListOf<DoubleArray>()
            val actions = mutableListOf<Action>()
            val rewards = mutableListOf<Double>()
            val logProbs = mutableListOf<Double>()

            var state = environment.reset()
            var done = false

//            while (!done) {
//                val (action, logProb) = selectAction(state, model)
//                val (nextState, reward) = environment.step(action)
//
//                states.add(state)
//                actions.add(action)
//                rewards.add(reward)
//                logProbs.add(logProb)
//
//                state = nextState
//                done = environment.isDone()
//            }
//
//            val returns = calculateReturns(rewards, gamma)
//            optimizeModel(states, actions, logProbs, returns, model, epsilon)
        }
    }
}
