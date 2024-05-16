//import main.kotlin.network.*
//import main.kotlin.train.Data
//
//class DeepQLearning(private val environment: Environment, learningRate: Double) {
//
//    private val model: NeuralNetwork = NeuralNetwork(learningRate, loss = MeanSquaredError)
//
//    init {
//        model.addLayer(Layer(environment.getState().size))
//        model.addLayer(Layer(32, LeakyReLU, NormalHeInitialization))
//        model.addLayer(Layer(environment.getActionSpaceSize(), Linear))
//        model.initialize()
//    }
//
//    fun train(episodes: Int, epsilon: Double, replayBufferSize: Int) {
//        val replayBuffer = ReplayBuffer(replayBufferSize)
//
//        for (episode in 0 until episodes) {
//            var state = environment.getState()
//            var done = false
//
//            while (!done) {
//                val action = pickAction(state, epsilon)
//                val nextState = environment.step(action)
//                val reward = environment.getReward()
//                done = environment.isTerminal()
//
//                replayBuffer.add(Experience(state, action, reward, nextState))
//
//                if (replayBuffer.size() >= replayBufferSize) {
//                    val experiences = replayBuffer.sample(batchSize)
//                    trainModel(experiences)
//                }
//
//                state = nextState
//            }
//
//            // Update epsilon for exploration decay (optional)
//            // epsilon *= 0.99
//            println("Episode: $episode")
//        }
//    }
//
//    private fun pickAction(state: DoubleArray, epsilon: Double): Int {
//        if (Math.random() < epsilon) {
//            return (0 until environment.getActionSpaceSize()).random()  // Random action
//        }
//        val qValues = model.predict(state)
//
//        return qValues.mapIndexed { index, d -> index to d }.maxByOrNull { it.second }!!.first
//    }
//
//    private fun trainModel(experiences: List<Experience>) {
//        val states = experiences.map { it.state }.toDoubleArray()
//        val actions = experiences.map { oneHot(it.action, environment.getActionSpaceSize()) }.toDoubleArray()
//        val rewards = experiences.map { it.reward }.toDoubleArray()
//        val nextStates = experiences.map { it.nextState }.toDoubleArray()
//
//        val targetQValues = nextStates.map { model.predict(it) }.toDoubleArray()
//        for (i in targetQValues.indices) {
//            targetQValues[i][experiences[i].action] = rewards[i] + gamma * targetQValues[i].max()
//        }
//
//        val data = Data()
//        model.fit(epochs = 1, data = data.setDataset(states, targetQValues))
//    }
//}
//
//
//
//data class Experience(val state: DoubleArray, val action: Int, val reward: Double, val nextState: DoubleArray)
//
//class ReplayBuffer(val maxSize: Int) {
//    private val buffer: MutableList<Experience> = mutableListOf()
//
//    fun size(): Int {
//        return buffer.size
//    }
//
//    fun add(experience: Experience) {
//        if (buffer.size >= maxSize) {
//            buffer.removeAt(0)
//        }
//        buffer.add(experience)
//    }
//
//    fun sample(size: Int): List<Experience> {
//        return buffer.shuffled().subList(0, Math.min(size, buffer.size))
//    }
//}
//
//fun oneHot(action: Int, size: Int): DoubleArray {
//    val array = DoubleArray(size)
//    array[action] = 1.0
//    return array
//}
