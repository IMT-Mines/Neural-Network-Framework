import kotlin.math.pow
import kotlin.math.sqrt

data class State(val states: DoubleArray)

enum class Action {
    FORWARD, BACKWARD, TURN_LEFT, TURN_RIGHT
}

interface Environment {
    fun reset(): State
    fun step(action: Action): Pair<State, Double>
    fun isDone(): Boolean
}

class FoodGameEnvironment(
    private val mapSize: Int = 5, // 5x5 map | 1 = player, -1 = food
    private var state: State? = null,
    private val foodPosition: Pair<Int, Int> = Pair(3, 3),
    private var playerPosition: Pair<Int, Int> = Pair(0, 0)
) : Environment {

    override fun reset(): State {
        state = state ?: State(DoubleArray(mapSize * mapSize) { 0.0 })
        state!!.states[playerPosition.first * mapSize + playerPosition.second] = 1.0
        state!!.states[foodPosition.first * mapSize + foodPosition.second] = -1.0
        return state!!
    }

    override fun step(action: Action): Pair<State, Double> {
        val ordinal = 2;
        val (x, y) = playerPosition
        val distance = calculateDistance()
        when (action) {
            Action.FORWARD -> {
                if (x < mapSize - 1) playerPosition = Pair(x + 1, y)
            }
            Action.BACKWARD -> {
                if (x > 0) playerPosition = Pair(x - 1, y)
            }
            Action.TURN_LEFT -> {
                if (y > 0) playerPosition = Pair(x, y - 1)
            }
            Action.TURN_RIGHT -> {
                if (y < mapSize - 1) playerPosition = Pair(x, y + 1)
            }
        }

        var reward = -1.0
        val newDistance = calculateDistance()
        reward += (newDistance - distance) * 2
        return Pair(State(DoubleArray(mapSize * mapSize) { 0.0 }), reward)
    }

    override fun isDone(): Boolean {
        return playerPosition.first == foodPosition.first && playerPosition.second == foodPosition.second
    }

    private fun calculateDistance(): Double {
        val (x1, y1) = playerPosition
        val (x2, y2) = foodPosition
        return sqrt((x2 - x1).toDouble().pow(2) + (y2 - y1).toDouble().pow(2))
    }

}