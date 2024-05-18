package main.kotlin.utils

import main.kotlin.train.Data
import java.util.concurrent.CompletableFuture
import kotlin.math.cos
import kotlin.math.ln
import kotlin.math.sqrt

class Utils {

    companion object {
        fun randomNormal(mean: Double, std: Double): Double {
            val u1 = Math.random()
            val u2 = Math.random()
            val z = sqrt(-2 * ln(u1)) * cos(2 * Math.PI * u2)
            return mean + z * std
        }

        fun awaitFutures(futures: List<CompletableFuture<*>>) {
            val completableFutures = CompletableFuture.allOf(*futures.toTypedArray())
            completableFutures.join()
        }

    }
}