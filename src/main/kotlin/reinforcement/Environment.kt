package main.kotlin.reinforcement

interface Environment<E: Enum<E>> {
    fun reset(): DoubleArray
    fun step(action: E): Pair<DoubleArray, Double>
    fun isDone(): Boolean
}
