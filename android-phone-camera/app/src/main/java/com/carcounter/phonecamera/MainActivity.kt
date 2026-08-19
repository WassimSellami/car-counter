package com.carcounter.phonecamera

import android.Manifest
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Bundle
import android.widget.Button
import android.widget.EditText
import android.widget.FrameLayout
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.security.crypto.EncryptedSharedPreferences
import androidx.security.crypto.MasterKey

class MainActivity : AppCompatActivity() {
    private lateinit var previewView: PreviewView
    private lateinit var url: EditText
    private lateinit var apiKey: EditText
    private lateinit var status: TextView
    private lateinit var cropOverlay: CropOverlay

    private val requestCamera = registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        if (granted) showPreview() else status.text = "Camera permission is required."
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val padding = (16 * resources.displayMetrics.density).toInt()
        val root = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setPadding(padding, padding, padding, padding) }
        // ImageAnalysis is 1920x1080 then rotated upright, so this exact 9:16
        // surface keeps crop-overlay coordinates aligned with uploaded pixels.
        val previewHeight = (resources.displayMetrics.widthPixels * 16f / 9f).toInt()
        val cameraArea = FrameLayout(this).apply { layoutParams = LinearLayout.LayoutParams(-1, previewHeight) }
        previewView = PreviewView(this).apply { scaleType = PreviewView.ScaleType.FIT_CENTER }
        cropOverlay = CropOverlay(this)
        cameraArea.addView(previewView, FrameLayout.LayoutParams(-1, -1))
        cameraArea.addView(cropOverlay, FrameLayout.LayoutParams(-1, -1))
        url = EditText(this).apply { hint = "Vast URL (http://IP:PORT)" }
        apiKey = EditText(this).apply { hint = "API key" }
        status = TextView(this)
        val start = Button(this).apply { text = "Start sending"; setOnClickListener { startSending() } }
        val stop = Button(this).apply { text = "Stop"; setOnClickListener { stopService(Intent(this@MainActivity, CameraUploadService::class.java)); status.text = "Stopped"; showPreview() } }
        root.addView(cameraArea)
        root.addView(url)
        root.addView(apiKey)
        root.addView(start)
        root.addView(stop)
        root.addView(status)
        setContentView(ScrollView(this).apply { addView(root) })

        val prefs = securePrefs()
        url.setText(prefs.getString("url", BuildConfig.DEFAULT_COUNTER_URL))
        apiKey.setText(prefs.getString("key", BuildConfig.DEFAULT_API_KEY))
        cropOverlay.setPoints(FloatArray(8) { index -> prefs.getFloat("point_$index", floatArrayOf(.1f, .1f, .9f, .1f, .9f, .9f, .1f, .9f)[index]) })
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) showPreview() else requestCamera.launch(Manifest.permission.CAMERA)
    }

    private fun startSending() {
        val baseUrl = url.text.toString().trim().trimEnd('/')
        val key = apiKey.text.toString().trim()
        if (!baseUrl.startsWith("http://") && !baseUrl.startsWith("https://")) { status.text = "Enter a full http(s) URL."; return }
        if (key.isBlank()) { status.text = "Enter the API key."; return }
        val previewPoints = cropOverlay.pointValues()
        val points = cameraImagePoints(previewPoints)
        securePrefs().edit().putString("url", baseUrl).putString("key", key)
            .apply { previewPoints.forEachIndexed { index, value -> putFloat("point_$index", value) } }.apply()
        // RootEncoder owns Camera2 while streaming; release the preview first.
        ProcessCameraProvider.getInstance(this).get().unbindAll()
        ContextCompat.startForegroundService(this, Intent(this, CameraUploadService::class.java)
            .putExtra(CameraUploadService.URL, baseUrl).putExtra(CameraUploadService.API_KEY, key)
            .putExtra(CameraUploadService.POINTS, points))
        cropOverlay.visibility = android.view.View.GONE
        status.text = "Sending in the background. Keep the phone mounted and powered."
    }

    private fun showPreview() {
        cropOverlay.visibility = android.view.View.VISIBLE
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            val provider = future.get()
            provider.unbindAll()
            provider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, Preview.Builder().build().also { it.surfaceProvider = previewView.surfaceProvider })
        }, ContextCompat.getMainExecutor(this))
    }

    private fun cameraImagePoints(overlayPoints: FloatArray): FloatArray {
        // PreviewView uses FIT_CENTER. Its surrounding view can therefore have
        // letterbox margins; convert overlay points into the visible 9:16 image.
        val viewWidth = previewView.width.toFloat()
        val viewHeight = previewView.height.toFloat()
        if (viewWidth == 0f || viewHeight == 0f) return overlayPoints
        val cameraAspect = 9f / 16f
        val viewAspect = viewWidth / viewHeight
        val imageWidth: Float
        val imageHeight: Float
        val left: Float
        val top: Float
        if (viewAspect > cameraAspect) {
            imageHeight = viewHeight
            imageWidth = imageHeight * cameraAspect
            left = (viewWidth - imageWidth) / 2f
            top = 0f
        } else {
            imageWidth = viewWidth
            imageHeight = imageWidth / cameraAspect
            left = 0f
            top = (viewHeight - imageHeight) / 2f
        }
        return FloatArray(8) { index ->
            val position = if (index % 2 == 0) overlayPoints[index] * viewWidth - left else overlayPoints[index] * viewHeight - top
            (position / if (index % 2 == 0) imageWidth else imageHeight).coerceIn(0f, 1f)
        }
    }

    private fun securePrefs() = EncryptedSharedPreferences.create(
        this, "settings", MasterKey.Builder(this).setKeyScheme(MasterKey.KeyScheme.AES256_GCM).build(),
        EncryptedSharedPreferences.PrefKeyEncryptionScheme.AES256_SIV,
        EncryptedSharedPreferences.PrefValueEncryptionScheme.AES256_GCM
    )
}
