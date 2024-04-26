package main.kotlin.train

import java.io.File
import java.util.*

class DataLoader {

    companion object {

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
                    data.add(features, 0.0)
                } else {
                    data.add(features, 1.0)
                }
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
                        data.add(features, 0.0)
                    }
                    "Iris-versicolor" -> {
                        data.add(features, 1.0)
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
                    data.add(features, 1.0)
                } else {
                    data.add(features, 0.0)
                }
            }
            return data
        }
    }
}

class Data {

    private val features = mutableListOf<DoubleArray>()
    private val labels = mutableListOf<Double>()

    fun add(features: DoubleArray, label: Double) {
        this.features.add(features)
        this.labels.add(label)
    }

    fun size(): Int {
        return features.size
    }

    fun get(index: Int): Pair<DoubleArray, Double> {
        return Pair(features[index], labels[index])
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
}
