package main.kotlin.utils

import main.kotlin.train.Data
import kotlin.math.cos
import kotlin.math.ln
import kotlin.math.sqrt

class Utils {

    companion object {

        /**
         * This function is used to normalize the data between 0 and 1 with Z-score normalization
         */
        fun normalizeZScore(data: Data) {
            val features = data.features
            val n = features[0].size
            val m = features.size
            val means = DoubleArray(n) { 0.0 }
            val stds = DoubleArray(n) { 0.0 }
            for (i in 0 until n) {
                var sum = 0.0
                for (j in 0 until m) {
                    sum += features[j][i]
                }
                means[i] = sum / m
            }
            for (i in 0 until n) {
                var sum = 0.0
                for (j in 0 until m) {
                    sum += (features[j][i] - means[i]) * (features[j][i] - means[i])
                }
                stds[i] = sqrt(sum / m)
            }
            for (i in 0 until n) {
                for (j in 0 until m) {
                    features[j][i] = (features[j][i] - means[i]) / stds[i]
                }

            }
        }

        fun randomNormal(mean: Double, std: Double): Double {
            val u1 = Math.random()
            val u2 = Math.random()
            val z = sqrt(-2 * ln(u1)) * cos(2 * Math.PI * u2)
            return mean + z * std
        }

    }
}