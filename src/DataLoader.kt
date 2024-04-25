import java.io.File
import java.util.*

class DataLoader {

    companion object {
        fun loadIonosphere(): Data {
            val data = Data()
            val file = File("ionosphere.data")
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
}
