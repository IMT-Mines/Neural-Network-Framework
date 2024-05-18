package main.kotlin.train

import java.io.File
import java.util.*

class DataLoader {

    companion object {

        fun loadSeeds(): Data {
            val data = Data()
            val file = File("src/main/resources/seeds.data")
            val scanner = Scanner(file)
            while (scanner.hasNextLine()) {
                val line = scanner.nextLine()
                val values = line.split(",")
                val features = DoubleArray(7) { 0.0 }
                for (i in 0..6) {
                    features[i] = values[i].toDouble()
                }
                val label = values[7]
                when (label) {
                    "1" -> {
                        data.add(features, doubleArrayOf(1.0, 0.0, 0.0))
                    }

                    "2" -> {
                        data.add(features, doubleArrayOf(0.0, 1.0, 0.0))
                    }

                    "3" -> {
                        data.add(features, doubleArrayOf(0.0, 0.0, 1.0))
                    }
                }
            }
            return data
        }

        fun loadSonar(): Data {
            val data = Data()
            val file = File("src/main/resources/sonar.data")
            val scanner = Scanner(file)
            while (scanner.hasNextLine()) {
                val line = scanner.nextLine()
                val values = line.split(",")
                val features = DoubleArray(60) { 0.0 }
                for (i in 0..59) {
                    features[i] = values[i].toDouble()
                }
                val label = values[60]
                if (label == "R") {
                    data.add(features, doubleArrayOf(0.0))
                } else {
                    data.add(features, doubleArrayOf(1.0))
                }
            }
            return data
        }

        fun loadDigits(): Data {
            val data = Data()
            val file = File("src/main/resources/digits.csv")
            val scanner = Scanner(file)
            scanner.nextLine()
            while (scanner.hasNextLine()) {
                val line = scanner.nextLine()
                val values = line.split(",")
                val features = DoubleArray(784) { 0.0 }
                for (i in 1..784) {
                    features[i - 1] = values[i].toDouble()
                }
                val label = values[0]
                val target = DoubleArray(10) { 0.0 }
                target[label.toInt()] = 1.0
                data.add(features, target)
            }
            return data
        }

        fun loadIris(): Data {
            val data = Data()
            val file = File("src/main/resources/iris.data")
            val scanner = Scanner(file)
            while (scanner.hasNextLine()) {
                val line = scanner.nextLine()
                if (line.isEmpty()) {
                    continue
                }
                val values = line.split(",")
                val features = DoubleArray(4) { 0.0 }
                for (i in 0..3) {
                    features[i] = values[i].toDouble()
                }
                val label = values[4]
                when (label) {
                    "Iris-setosa" -> {
                        data.add(features, doubleArrayOf(1.0, 0.0, 0.0))
                    }

                    "Iris-versicolor" -> {
                        data.add(features, doubleArrayOf(0.0, 1.0, 0.0))
                    }

                    "Iris-virginica" -> {
                        data.add(features, doubleArrayOf(0.0, 0.0, 1.0))
                    }
                }
            }
            return data
        }

        fun loadIonosphere(): Data {
            val data = Data()
            val file = File("src/main/resources/ionosphere.data")
            val scanner = Scanner(file)
            while (scanner.hasNextLine()) {
                val line = scanner.nextLine()
                val values = line.split(",")
                val features = DoubleArray(33) { 0.0 }
                for (i in 0..<33) {
                    features[i] = values[i].toDouble()
                }
                val label = values[34]
                if (label == "g") {
                    data.add(features, doubleArrayOf(1.0))
                } else {
                    data.add(features, doubleArrayOf(0.0))
                }
            }
            return data
        }
    }
}

class Data {

    val features = mutableListOf<DoubleArray>()
    private val labels = mutableListOf<DoubleArray>()

    fun add(features: DoubleArray, label: DoubleArray) {
        this.features.add(features)
        this.labels.add(label)
    }

    fun size(): Int {
        return features.size
    }

    fun get(index: Int): Pair<DoubleArray, DoubleArray> {
        return Pair(features[index], labels[index])
    }

    fun normalizeMinMaxFeatures(): Pair<DoubleArray, DoubleArray> {
        val mins = DoubleArray(features[0].size) { Double.MAX_VALUE }
        val maxs = DoubleArray(features[0].size) { Double.MIN_VALUE }
        for (i in 0 until features.size) {
            for (j in 0 until features[0].size) {
                if (features[i][j] < mins[j]) {
                    mins[j] = features[i][j]
                }
                if (features[i][j] > maxs[j]) {
                    maxs[j] = features[i][j]
                }
            }
        }

        for (i in 0 until features.size) {
            for (j in 0 until features[0].size) {
                features[i][j] = (features[i][j] - mins[j]) / (maxs[j] - mins[j])
            }
        }
        return Pair(mins, maxs)
    }

    companion object {
        fun normalizeNewData(newFeatures: Array<DoubleArray>, mins: DoubleArray, maxs: DoubleArray) {
            for (i in newFeatures.indices) {
                for (j in 0 until newFeatures[0].size) {
                    newFeatures[i][j] = (newFeatures[i][j] - mins[j]) / (maxs[j] - mins[j])
                }
            }
        }
    }

    fun split(ratio: Double): Pair<Data, Data> {
        val trainData = Data()
        val testData = Data()
        val shuffled = features.zip(labels).shuffled()
        val splitIndex = (ratio * features.size).toInt()
        val trainFeatures = shuffled.subList(0, splitIndex).map { it.first }.toMutableList()
        val trainLabels = shuffled.subList(0, splitIndex).map { it.second }.toMutableList()
        val testFeatures = shuffled.subList(splitIndex, features.size).map { it.first }.toMutableList()
        val testLabels = shuffled.subList(splitIndex, features.size).map { it.second }.toMutableList()
        for (i in 0..<trainFeatures.size) {
            trainData.add(trainFeatures[i], trainLabels[i])
        }
        for (i in 0..<testFeatures.size) {
            testData.add(testFeatures[i], testLabels[i])
        }
        return Pair(trainData, testData)
    }

    fun shuffle() {
        val zipped = features.zip(labels).shuffled()
        features.clear()
        labels.clear()
        for (pair in zipped) {
            features.add(pair.first)
            labels.add(pair.second)
        }
    }

    fun setDataset(features: List<DoubleArray>, labels: List<DoubleArray>) {
        this.features.clear()
        this.labels.clear()
        this.features.addAll(features)
        this.labels.addAll(labels)
    }

    override fun toString(): String {
        val stringBuilder = StringBuilder()
        for (i in 0..<features.size) {
            stringBuilder.append(features[i].contentToString())
            stringBuilder.append(" -> ")
            stringBuilder.append(labels[i].contentToString())
            stringBuilder.append("\n")
        }
        return stringBuilder.toString()
    }
}
