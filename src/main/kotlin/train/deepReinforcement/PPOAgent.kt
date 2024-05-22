//package train.deepReinforcement
//
//import Action
//import Environment
//import State
//import main.kotlin.network.NeuralNetwork
//
//
//class PPOAgent(private val model: NeuralNetwork) {
//    fun train(env: Environment, epochs: Int, batchSize: Int = 1, gamma: Double = 0.99, epsilon: Double = 0.2) {
//        for (epoch in 0 until epochs) {
//            val states = mutableListOf<State>()
//            val actions = mutableListOf<Action>()
//            val rewards = mutableListOf<Double>()
//            val logProbs = mutableListOf<Double>()
//
//            var state = env.reset()
//            var done = false
//
//            while (!done) {
//                val (action, logProb) = selectAction(state, model)
//                val (nextState, reward) = env.step(action)
//
//                states.add(state)
//                actions.add(action)
//                rewards.add(reward)
//                logProbs.add(logProb)
//
//                state = nextState
//                done = env.isTerminal()
//            }
//
//            val returns = calculateReturns(rewards, gamma)
//            optimizeModel(states, actions, logProbs, returns, model, epsilon)
//        }
//    }
//
//    fun calculateReturns(rewards: List<Double>, gamma: Double = 0.99): List<Double> {
//        val returns = MutableList(rewards.size) { 0.0 }
//        var runningTotal = 0.0
//
//        for (i in rewards.indices.reversed()) {
//            runningTotal = rewards[i] + gamma * runningTotal
//            returns[i] = runningTotal
//        }
//
//        return returns
//    }
//
//    fun selectAction(state: State, model: NeuralNetwork): Pair<Action, Double> {
//        val stateInput = state.toInputArray()
//        val actionProbabilities = model.predict(stateInput)
//
//        val actionIndex = sampleAction(actionProbabilities)
//        val logProbability = kotlin.math.log(actionProbabilities[actionIndex])
//
//        return Pair(Action.entries[actionIndex], logProbability)
//    }
//
//    private fun State.toInputArray(): DoubleArray {
//        // Convertir l'état en un array d'entrées pour le modèle
//        return doubleArrayOf(position.first, position.second)
//    }
//
//    private fun sampleAction(probabilities: DoubleArray): Int {
//        // Échantillonner une action basée sur les probabilités
//        val cumulativeDistribution = probabilities.scan(0.0, Double::plus)
//        val randomValue = Math.random()
//        return cumulativeDistribution.indexOfFirst { it > randomValue } - 1
//    }
//
//    private fun optimizeModel(
//        states: List<State>,
//        actions: List<Action>,
//        oldLogProbs: List<Double>,
//        returns: List<Double>,
//        model: NeuralNetwork,
//        epsilon: Double = 0.2
//    ) {
//        val advantages = returns.toDoubleArray() // Placeholder for simplicity
//
//        for (i in states.indices) {
//            val stateInput = states[i].toInputArray()
//            val actionIndex = actions[i].ordinal
//            val oldLogProb = oldLogProbs[i]
//
//            val newActionProbabilities = model.predict(stateInput)
//            val newLogProb = kotlin.math.log(newActionProbabilities[actionIndex])
//
//            val ratio = kotlin.math.exp(newLogProb - oldLogProb)
//            val surr1 = ratio * advantages[i]
//            val surr2 = ratio.coerceIn(1 - epsilon, 1 + epsilon) * advantages[i]
//
//            val loss = -kotlin.math.min(surr1, surr2)
//
//            // Calcul des gradients et mise à jour des paramètres
//            val lossDerivative = doubleArrayOf(loss)
//            model.optimizer.minimize(model, lossDerivative)
//        }
//    }
//
//}
