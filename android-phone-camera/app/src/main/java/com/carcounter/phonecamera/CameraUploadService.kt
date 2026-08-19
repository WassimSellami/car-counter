package com.carcounter.phonecamera

import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Intent
import android.net.Uri
import android.os.IBinder
import android.util.Log
import androidx.core.app.NotificationCompat
import com.pedro.common.ConnectChecker
import com.pedro.encoder.input.sources.audio.NoAudioSource
import com.pedro.encoder.input.sources.video.Camera2Source
import com.pedro.library.rtmp.RtmpStream
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import java.net.InetSocketAddress
import java.net.Socket
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit

/** Hardware H.264 camera publisher. Vast performs the perspective crop. */
class CameraUploadService : Service(), ConnectChecker {
    companion object {
        const val URL = "url"
        const val API_KEY = "apiKey"
        const val POINTS = "points"
        private const val STOP = "com.carcounter.phonecamera.STOP"
        private const val CHANNEL = "capture"
        private const val TAG = "CarCounterUpload"
    }

    private val worker = Executors.newSingleThreadExecutor()
    private val client = OkHttpClient.Builder().connectTimeout(15, TimeUnit.SECONDS).readTimeout(15, TimeUnit.SECONDS).build()
    private var stream: RtmpStream? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == STOP) {
            stopSelf()
            return START_NOT_STICKY
        }
        val baseUrl = intent?.getStringExtra(URL)?.trimEnd('/') ?: return START_NOT_STICKY
        val apiKey = intent.getStringExtra(API_KEY) ?: return START_NOT_STICKY
        val points = intent.getFloatArrayExtra(POINTS)?.takeIf { it.size == 8 } ?: return START_NOT_STICKY
        startForeground(1, notification("Preparing H.264 stream…"))
        worker.execute {
            try {
                saveCrop(baseUrl, apiKey, points)
                val rtmpUrl = rtmpUrl(baseUrl, apiKey)
                checkRtmpTcp(rtmpUrl)
                val publisher = RtmpStream(this, this, Camera2Source(this), NoAudioSource())
                if (!publisher.prepareVideo(1920, 1080, 6_000_000, 30, 2, 90) || !publisher.prepareAudio(32_000, false, 64_000)) {
                    report("Could not prepare H.264 stream")
                    stopSelf()
                    return@execute
                }
                stream = publisher
                publisher.startStream(rtmpUrl)
                report("Connecting H.264 stream…")
            } catch (error: Exception) {
                report("Start failed: ${error.message ?: error.javaClass.simpleName}")
                stopSelf()
            }
        }
        return START_NOT_STICKY
    }

    private fun saveCrop(baseUrl: String, apiKey: String, points: FloatArray) {
        val json = "{\"points\":[${points.joinToString(",")}]}"
        val request = Request.Builder().url("$baseUrl/phone-crop")
            .header("X-API-Key", apiKey)
            .post(json.toRequestBody("application/json".toMediaType())).build()
        client.newCall(request).execute().use { response ->
            check(response.isSuccessful) { "Crop setup returned HTTP ${response.code}" }
        }
    }

    private fun rtmpUrl(baseUrl: String, apiKey: String): String {
        val host = Uri.parse(baseUrl).host ?: error("Invalid Vast URL")
        return "rtmp://phone:${Uri.encode(apiKey)}@$host:${BuildConfig.RTMP_PORT}/phone"
    }

    private fun checkRtmpTcp(url: String) {
        val target = Uri.parse(url)
        Socket().use { socket -> socket.connect(InetSocketAddress(target.host, target.port), 5_000) }
        Log.i(TAG, "RTMP TCP reachable: ${target.host}:${target.port}")
    }

    override fun onConnectionStarted(url: String) = report("Connecting…")
    override fun onConnectionSuccess() = report("Streaming 1080p H.264 at 30 FPS")
    override fun onConnectionFailed(reason: String) = report("Stream failed: $reason")
    override fun onDisconnect() = report("Stream disconnected")
    override fun onAuthError() = report("RTMP authentication failed")
    override fun onAuthSuccess() = report("RTMP authentication accepted")

    private fun report(message: String) {
        Log.i(TAG, message)
        getSystemService(NotificationManager::class.java).notify(1, notification(message))
    }

    private fun notification(message: String): android.app.Notification {
        val manager = getSystemService(NotificationManager::class.java)
        manager.createNotificationChannel(NotificationChannel(CHANNEL, "Camera stream", NotificationManager.IMPORTANCE_LOW))
        val stop = PendingIntent.getService(this, 0, Intent(this, CameraUploadService::class.java).setAction(STOP), PendingIntent.FLAG_IMMUTABLE)
        return NotificationCompat.Builder(this, CHANNEL).setSmallIcon(android.R.drawable.presence_video_online)
            .setContentTitle("Car counter camera active").setContentText(message).addAction(0, "Stop", stop).build()
    }

    override fun onDestroy() {
        stream?.let { if (it.isStreaming) it.stopStream() }
        client.dispatcher.executorService.shutdown()
        worker.shutdown()
        super.onDestroy()
    }

    override fun onBind(intent: Intent?): IBinder? = null
}
