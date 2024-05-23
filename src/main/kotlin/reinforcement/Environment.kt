package main.kotlin.reinforcement

enum class Action {
    FORWARD, BACKWARD, TURN_LEFT, TURN_RIGHT
}

interface Environment {
    fun reset(): DoubleArray
    fun step(action: Action): Pair<DoubleArray, Double>
    fun isDone(): Boolean
}
