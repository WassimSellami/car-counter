package com.carcounter.phonecamera

import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Intent
import android.hardware.camera2.CaptureRequest
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.YuvImage
import android.os.IBinder
import android.util.Log
import android.util.Range
import android.util.Size
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.camera2.interop.Camera2Interop
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.NotificationCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleService
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.MultipartBody
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import java.io.ByteArrayOutputStream
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean

class CameraUploadService : LifecycleService() {
    companion object {
        const val URL = "url"
        const val API_KEY = "apiKey"
        const val POINTS = "points"
        private const val STOP = "com.carcounter.phonecamera.STOP"
        private const val CHANNEL = "capture"
        private const val TAG = "CarCounterUpload"
        private const val BATCH_SIZE = 10
        private const val SEND_WIDTH = 1080
        private const val JPEG_QUALITY = 92
    }
    private val cameraExecutor = Executors.newSingleThreadExecutor()
    private val networkExecutor = Executors.newSingleThreadExecutor()
    private val client = OkHttpClient.Builder().connectTimeout(15, TimeUnit.SECONDS).readTimeout(30, TimeUnit.SECONDS).build()
    private val uploading = AtomicBoolean(false)
    private val frames = mutableListOf<ByteArray>()
    private lateinit var endpoint: String
    private lateinit var apiKey: String
    private var points = floatArrayOf(0f, 0f, 1f, 0f, 1f, 1f, 0f, 1f)

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        // LifecycleService advances its lifecycle through Service.onStartCommand().
        // Without this call CameraX attaches the analyser, then immediately closes it.
        super.onStartCommand(intent, flags, startId)
        if (intent?.action == STOP) {
            stopSelf()
            return Service.START_NOT_STICKY
        }
        endpoint = intent?.getStringExtra(URL)?.trimEnd('/')?.plus("/ingest-batch") ?: return Service.START_NOT_STICKY
        apiKey = intent.getStringExtra(API_KEY) ?: return Service.START_NOT_STICKY
        points = intent.getFloatArrayExtra(POINTS)?.takeIf { it.size == 8 } ?: points
        startForeground(1, notification("Starting camera…"))
        startCamera()
        return Service.START_STICKY
    }

    private fun startCamera() {
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            val builder = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setTargetResolution(Size(1920, 1080))
            Camera2Interop.Extender(builder).setCaptureRequestOption(CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, Range(30, 30))
            val analysis = builder.build()
            analysis.setAnalyzer(cameraExecutor) { image ->
                try {
                    addFrame(image.toJpeg())
                } catch (error: Exception) {
                    Log.e(TAG, "Frame conversion failed", error)
                    report("Camera frame failed: ${error.javaClass.simpleName}")
                } finally {
                    image.close()
                }
            }
            try {
                future.get().unbindAll()
                future.get().bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, analysis)
                Log.i(TAG, "Camera analyser is active")
                report("Camera ready; collecting frames")
            } catch (error: Exception) {
                Log.e(TAG, "Could not start camera", error)
                report("Camera failed: ${error.javaClass.simpleName}")
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun addFrame(jpeg: ByteArray) = synchronized(frames) {
        frames += jpeg
        if (frames.size < BATCH_SIZE) return
        // Vast is slower than the camera sometimes. Drop this batch instead of
        // retaining stale images (or allowing an unbounded in-memory queue).
        if (!uploading.compareAndSet(false, true)) {
            frames.clear()
            return
        }
        val batch = frames.toList()
        frames.clear()
        Log.i(TAG, "Batch ready: ${batch.size} frames")
        networkExecutor.execute {
            try {
                upload(batch)
            } catch (error: Exception) {
                Log.e(TAG, "Upload failed", error)
                report("Upload failed: ${error.message ?: error.javaClass.simpleName}")
            } finally {
                uploading.set(false)
            }
        }
    }

    private fun upload(batch: List<ByteArray>) {
        val body = MultipartBody.Builder().setType(MultipartBody.FORM).apply {
            batch.forEach { addFormDataPart("frames", "frame.jpg", it.toRequestBody("image/jpeg".toMediaType())) }
        }.build()
        Log.i(TAG, "Sending ${batch.size} frames")
        client.newCall(Request.Builder().url(endpoint).header("X-API-Key", apiKey).post(body).build()).execute().use { response ->
            val message = "Upload ${response.code} ${response.message}"
            if (response.isSuccessful) Log.i(TAG, message) else Log.w(TAG, message)
            report(message)
        }
    }

    private fun ImageProxy.toJpeg(): ByteArray {
        val nv21 = ByteArray(width * height * 3 / 2)
        val y = planes[0]; var offset = 0
        for (row in 0 until height) { y.buffer.position(row * y.rowStride); y.buffer.get(nv21, offset, width); offset += width }
        val u = planes[1]; val v = planes[2]
        for (row in 0 until height / 2) for (col in 0 until width / 2) {
            v.buffer.position(row * v.rowStride + col * v.pixelStride); nv21[offset++] = v.buffer.get()
            u.buffer.position(row * u.rowStride + col * u.pixelStride); nv21[offset++] = u.buffer.get()
        }
        val raw = ByteArrayOutputStream()
        YuvImage(nv21, ImageFormat.NV21, width, height, null).compressToJpeg(Rect(0, 0, width, height), 80, raw)
        val bitmap = BitmapFactory.decodeByteArray(raw.toByteArray(), 0, raw.size())
        val upright = if (imageInfo.rotationDegrees == 0) bitmap else Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, Matrix().apply { postRotate(imageInfo.rotationDegrees.toFloat()) }, true)
        val source = FloatArray(8) { index -> if (index % 2 == 0) points[index] * upright.width else points[index] * upright.height }
        val outputWidth = ((distance(source, 0, 2) + distance(source, 6, 4)) / 2).toInt().coerceAtLeast(2)
        val outputHeight = ((distance(source, 0, 6) + distance(source, 2, 4)) / 2).toInt().coerceAtLeast(2)
        val destination = floatArrayOf(0f, 0f, outputWidth.toFloat(), 0f, outputWidth.toFloat(), outputHeight.toFloat(), 0f, outputHeight.toFloat())
        val projection = Matrix().apply { setPolyToPoly(source, 0, destination, 0, 4) }
        val cropped = Bitmap.createBitmap(outputWidth, outputHeight, Bitmap.Config.ARGB_8888)
        android.graphics.Canvas(cropped).drawBitmap(upright, projection, Paint(Paint.FILTER_BITMAP_FLAG))
        val scaled = if (cropped.width > SEND_WIDTH) Bitmap.createScaledBitmap(cropped, SEND_WIDTH, cropped.height * SEND_WIDTH / cropped.width, true) else cropped
        return ByteArrayOutputStream().use { out -> scaled.compress(Bitmap.CompressFormat.JPEG, JPEG_QUALITY, out); out.toByteArray() }
    }

    private fun distance(points: FloatArray, first: Int, second: Int): Float {
        val dx = points[first] - points[second]
        val dy = points[first + 1] - points[second + 1]
        return kotlin.math.hypot(dx, dy)
    }

    private fun report(message: String) {
        getSystemService(NotificationManager::class.java).notify(1, notification(message))
    }

    private fun notification(message: String) : android.app.Notification {
        val manager = getSystemService(NotificationManager::class.java)
        manager.createNotificationChannel(NotificationChannel(CHANNEL, "Camera upload", NotificationManager.IMPORTANCE_LOW))
        val stop = PendingIntent.getService(this, 0, Intent(this, CameraUploadService::class.java).setAction(STOP), PendingIntent.FLAG_IMMUTABLE)
        return NotificationCompat.Builder(this, CHANNEL).setSmallIcon(android.R.drawable.presence_video_online).setContentTitle("Car counter camera active").setContentText(message).addAction(0, "Stop", stop).build()
    }

    override fun onDestroy() { client.dispatcher.executorService.shutdown(); cameraExecutor.shutdown(); networkExecutor.shutdown(); super.onDestroy() }
}
