package main.kotlin.charts

import org.jetbrains.kotlinx.dataframe.api.dataFrameOf
import org.jetbrains.kotlinx.kandy.dsl.plot
import org.jetbrains.kotlinx.kandy.letsplot.export.save
import org.jetbrains.kotlinx.kandy.letsplot.feature.layout
import org.jetbrains.kotlinx.kandy.letsplot.layers.line
import org.jetbrains.kotlinx.kandy.util.color.Color

class Chart {


    companion object {
        fun lineChart(
            datas: MutableMap<Int, Double>, title: String, xLabel: String, yLabel: String, chartColor: Color = Color.BLUE) {
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
            }.save("$title.png")
        }
    }
}