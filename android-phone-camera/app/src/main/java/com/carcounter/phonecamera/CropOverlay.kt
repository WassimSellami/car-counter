package com.carcounter.phonecamera

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.view.MotionEvent
import android.view.View

/** Four-point perspective selector. Values are stored as x/y fractions of the preview. */
class CropOverlay(context: Context) : View(context) {
    private val shade = Paint().apply { color = 0x88000000.toInt() }
    private val line = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.WHITE; style = Paint.Style.STROKE; strokeWidth = 4f }
    private val handle = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.WHITE }
    private var points = floatArrayOf(.1f, .1f, .9f, .1f, .9f, .9f, .1f, .9f)
    private var activePoint = -1

    fun pointValues() = points.copyOf()
    fun setPoints(values: FloatArray) { if (values.size == 8) { points = values.copyOf(); invalidate() } }

    override fun onDraw(canvas: Canvas) {
        val path = android.graphics.Path()
        points.indices.step(2).forEachIndexed { index, offset ->
            val x = points[offset] * width
            val y = points[offset + 1] * height
            if (index == 0) path.moveTo(x, y) else path.lineTo(x, y)
        }
        path.close()
        canvas.drawPath(path, line)
        points.indices.step(2).forEach { offset -> canvas.drawCircle(points[offset] * width, points[offset + 1] * height, 14f, handle) }
    }

    override fun onTouchEvent(event: MotionEvent): Boolean {
        if (width == 0 || height == 0) return false
        val x = (event.x / width).coerceIn(0f, 1f)
        val y = (event.y / height).coerceIn(0f, 1f)
        when (event.actionMasked) {
            MotionEvent.ACTION_DOWN -> activePoint = nearestPoint(x, y)
            MotionEvent.ACTION_MOVE -> if (activePoint >= 0) {
                points[activePoint * 2] = x
                points[activePoint * 2 + 1] = y
                invalidate()
            }
            MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> activePoint = -1
        }
        return true
    }

    private fun nearestPoint(x: Float, y: Float): Int {
        return (0..3).minBy { index ->
            val cx = points[index * 2]
            val cy = points[index * 2 + 1]
            (x - cx) * (x - cx) + (y - cy) * (y - cy)
        }
    }
}
