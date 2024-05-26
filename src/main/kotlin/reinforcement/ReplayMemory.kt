package main.kotlin.reinforcement

class Experience(val state: DoubleArray, val action: Int, val nextState: DoubleArray, val reward: Double)

class ReplayMemory(private val maxMemory: Int) {
    private val memory = mutableListOf<Experience>()

    fun add(experience: Experience) {
        if (memory.size >= maxMemory) {
            memory.removeAt(0)
        }
        memory.add(experience)
    }

    fun sample(batchSize: Int): List<Experience> {
        return memory.shuffled().take(batchSize)
    }

    fun size(): Int {
        return memory.size
    }
}