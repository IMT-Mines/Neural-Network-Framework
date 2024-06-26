package main.kotlin.train.deepReinforcement

import main.kotlin.reinforcement.Environment
import kotlin.math.pow
import kotlin.math.sqrt

enum class FoodGameAction {
    FORWARD, BACKWARD, TURN_LEFT, TURN_RIGHT
}

class FoodGameEnvironment(
    private val mapSize: Int = 5, // 5x5 map | 1 = player, -1 = food
    private var state: DoubleArray = DoubleArray(mapSize * mapSize) { 0.0 },
    private var foodPosition: Pair<Int, Int> = Pair(mapSize - 1, mapSize - 1),
    private var playerPosition: Pair<Int, Int> = Pair(0, 0)
) : Environment<FoodGameAction> {

    override fun reset(): DoubleArray {
        playerPosition = Pair(0, 0)
//        foodPosition = Pair((0 until mapSize).random(), (0 until mapSize).random())
        foodPosition = Pair(mapSize - 1, mapSize - 1)
        state = DoubleArray(mapSize * mapSize) { 0.0 }
        state[playerPosition.first * mapSize + playerPosition.second] = 1.0
        state[foodPosition.first * mapSize + foodPosition.second] = -1.0
        return state
    }

    override fun step(actionIndex: Int): Pair<DoubleArray, Double> {
        val (x, y) = playerPosition
        state[playerPosition.first * mapSize + playerPosition.second] = 0.0
        val distance = calculateDistance()
        val action = FoodGameAction.entries[actionIndex]
        when (action) {
            FoodGameAction.FORWARD -> {
                if (x < mapSize - 1) playerPosition = Pair(x + 1, y)
            }
            FoodGameAction.BACKWARD -> {
                if (x > 0) playerPosition = Pair(x - 1, y)
            }
            FoodGameAction.TURN_LEFT -> {
                if (y > 0) playerPosition = Pair(x, y - 1)
            }
            FoodGameAction.TURN_RIGHT -> {
                if (y < mapSize - 1) playerPosition = Pair(x, y + 1)
            }
        }

        state[playerPosition.first * mapSize + playerPosition.second] = 1.0

        var reward = 0.0
        val newDistance = calculateDistance()
        reward += (distance - newDistance) * 2
        if (playerPosition.first == foodPosition.first && playerPosition.second == foodPosition.second) {
            reward += 100.0
        }

        return Pair(state, reward)
    }

    override fun isDone(): Boolean {
        return playerPosition.first == foodPosition.first && playerPosition.second == foodPosition.second
    }

    override fun randomAction(): Int {
        return (0 until FoodGameAction.entries.size).random()
    }

    private fun calculateDistance(): Double {
        val (x1, y1) = playerPosition
        val (x2, y2) = foodPosition
        return sqrt((x2 - x1).toDouble().pow(2) + (y2 - y1).toDouble().pow(2))
    }

    fun printState() {
        for (i in 0 until mapSize) {
            for (j in 0 until mapSize) {
                val index = i * mapSize + j
                if (state[index] == 1.0) {
                    print("P ")
                } else if (state[index] == -1.0) {
                    print("F ")
                } else {
                    print(". ")
                }
            }
            println()
        }
    }

    override fun toString(): String {
        var str = ""
        for (i in 0 until mapSize) {
            for (j in 0 until mapSize) {
                val index = i * mapSize + j
                str += when {
                    state[index] == 1.0 -> "P "
                    state[index] == -1.0 -> "F "
                    else -> ". "
                }
            }
            str += "\n"
        }
        return str
    }
}
