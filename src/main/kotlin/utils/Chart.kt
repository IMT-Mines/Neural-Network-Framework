package main.kotlin.utils

import org.jetbrains.kotlinx.dataframe.api.dataFrameOf
import org.jetbrains.kotlinx.kandy.dsl.plot
import org.jetbrains.kotlinx.kandy.letsplot.export.save
import org.jetbrains.kotlinx.kandy.letsplot.feature.layout
import org.jetbrains.kotlinx.kandy.letsplot.layers.line
import org.jetbrains.kotlinx.kandy.util.color.Color

class Chart {


    companion object {
        fun lineChart(
            datas: MutableMap<Int, Double>,
            title: String,
            xLabel: String,
            yLabel: String,
            chartColor: Color = Color.BLUE,
            path: String = "src/main/resources/plots/"
        ) {
            val x = datas.keys.toList()
            val y = datas.values.toList()
            val dataFrameOf = dataFrameOf(xLabel to x.toList(), yLabel to y.toList())
            dataFrameOf.plot {
                line {
                    x(xLabel)
                    y(yLabel)
                    color = chartColor
                }
                layout.title = title
            }.save("$title.png", path = path)
        }

        fun multiLineChart(
            datas: List<MutableMap<Int, Double>>,
            title: String,
            iterationKey: String,
            valueKey: String,
            weightKey: String,
            path: String = "src/main/resources/plots/"
        ) {
            val dataset = mutableMapOf<String, List<Any>>()

            for ((index, data) in datas.withIndex()) {
                val prefix = "$weightKey $index"
                dataset[iterationKey] = data.keys.toList() + (dataset[iterationKey] ?: emptyList())
                dataset[valueKey] = data.values.toList() + (dataset[valueKey] ?: emptyList())
                dataset[weightKey] = (0 until data.size).map { "$prefix" } + (dataset[weightKey] ?: emptyList())
            }

            dataset.plot {
                groupBy(weightKey) {
                    line {
                        x(iterationKey)
                        y(valueKey)
                        color(weightKey)
                    }
                }
                layout.title = title
                layout.size = 1200 to 800
            }.save("$title.png", path = path)
        }
    }
}
