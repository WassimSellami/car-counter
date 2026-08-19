# Phone Camera Sender (Android)

Native Kotlin app that replaces `cloud_camera_sender.py`: it requests 1080p/30 FPS from the rear camera, applies a four-point perspective crop, and uploads high-quality 1080px-wide JPEG groups to Vast's existing `/ingest-batch` route.

## Build and use

1. Open this `android-phone-camera` folder in Android Studio and allow it to install the Android SDK components it requests.
2. Run it on an Android phone (Android 8+), grant Camera permission, and enter:
   - Vast URL: `http://VAST_IP:PUBLIC_8000_PORT`
   - API key: the same `CLOUD_INFERENCE_API_KEY` configured on Vast.
3. Mount the phone, use the rear camera, frame the road in the preview, and tap **Start sending**.

For a new installation, the fields are automatically filled from the repository's ignored parent [`.env`](../.env) file when you build the APK: `COUNTER_PUBLIC_URL` and `CLOUD_INFERENCE_API_KEY`. After starting once, the app stores them encrypted on the phone, so you do not need to re-enter them on later launches. Rebuild the APK after changing either value.

The app continues through a foreground service; Android displays an ongoing notification while it is active. It deliberately drops queued frames when Vast or the network is slow, so counting operates on current traffic.

## Security

This initial version permits HTTP only because the current Vast server exposes plain HTTP. That exposes images and the API key on the network. Use it only on a trusted network while testing; switch the Vast endpoint to HTTPS and then remove `android:usesCleartextTraffic="true"` before real deployment.

The build-time defaults are embedded in the APK, so never distribute this APK or commit a real `.env` file. They are a convenience for your own phone only.
