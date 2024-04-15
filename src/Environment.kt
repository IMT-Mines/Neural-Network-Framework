class Environment {

    companion object {
        var map = Array(3) { Array(3) { 0 } }
        var agentPosition = Pair(0, 0)
    }

    init {
        val x = (0..9).random()
        val y = (0..9).random()
        map[x][y] = 1
        agentPosition = Pair(x, y)
        println(map.joinToString("\n") { it.joinToString(" ") })
    }

    fun getState(): DoubleArray {
        return doubleArrayOf(agentPosition.first.toDouble(), agentPosition.second.toDouble())
    }

    fun action(action: Int) {
        val currentX = agentPosition.first
        val currentY = agentPosition.second

        when (action) {
            0 -> {
                if (currentX > 0) {
                    agentPosition = Pair(currentX - 1, currentY)
                }
            }

            1 -> {
                if (currentX < 63) {
                    agentPosition = Pair(currentX + 1, currentY)
                }
            }

            2 -> {
                if (currentY > 0) {
                    agentPosition = Pair(currentX, currentY - 1)
                }
            }

            3 -> {
                if (currentY < 63) {
                    agentPosition = Pair(currentX, currentY + 1)
                }
            }
        }
    }

    fun getReward(): Int {
        val currentX = agentPosition.first
        val currentY = agentPosition.second

        if (currentX == 2 && currentY == 2) {
            return 1
        } else if (currentX == 1 && currentY == 1) {
            return -1
        } else {
            return 0
        }
    }
}