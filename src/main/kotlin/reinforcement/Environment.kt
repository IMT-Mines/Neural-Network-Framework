package main.kotlin.reinforcement

interface Environment<E: Enum<E>> {
    fun reset(): DoubleArray
    fun step(actionIndex: Int): Pair<DoubleArray, Double>
    fun isDone(): Boolean
    fun randomAction(): Int
}
