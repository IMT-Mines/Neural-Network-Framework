class Environment(private val mapSize: Int) {

    private var map: Array<Array<Int>> = Array(mapSize) { Array(mapSize) { 0 } }
    private var agentPosition: Pair<Int, Int> = Pair(mapSize - 1, 0)

    fun getActionSpaceSize(): Int {
        return 4
    }

    fun getState(): DoubleArray {
        return doubleArrayOf(agentPosition.first.toDouble(), agentPosition.second.toDouble())
    }

    fun action(action: Int) {
        val currentX = agentPosition.first
        val currentY = agentPosition.second
        map[currentX][currentY] = 0

        when (action) {
            0 -> {
                if (currentX > 0) {
                    agentPosition = Pair(currentX - 1, currentY)
                }
            }
            1 -> {
                if (currentX < mapSize - 1) {
                    agentPosition = Pair(currentX + 1, currentY)
                }
            }
            2 -> {
                if (currentY > 0) {
                    agentPosition = Pair(currentX, currentY - 1)
                }
            }
            3 -> {
                if (currentY < mapSize - 1) {
                    agentPosition = Pair(currentX, currentY + 1)
                }
            }
        }
        map[agentPosition.first][agentPosition.second] = 1
    }

    fun getReward(): Double {
        val currentX = agentPosition.first
        val currentY = agentPosition.second

        return if (currentX == 2 && currentY == 2) {
            1.0
        } else if (currentX == 1 && currentY == 1) {
            -1.0
        } else {
            0.0
        }
    }

    fun isTerminal(): Boolean {
        val currentX = agentPosition.first
        val currentY = agentPosition.second
        return currentX == 2 && currentY == 2 || currentX == 1 && currentY == 1
    }

    fun step(action: Int): Pair<DoubleArray, Double> {
        action(action)
        val nextState = getState()
        val reward = getReward()
        return Pair(nextState, reward)
    }

    override fun toString(): String {
        return map.joinToString("\n") { it.joinToString(" ") }
    }
}