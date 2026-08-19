# Phone Camera Sender (Android)

Native Kotlin app that replaces `cloud_camera_sender.py`: it encodes the rear camera with hardware H.264 at 1080p/30 FPS and publishes it to Vast over RTMP. Vast performs the four-point perspective crop, then submits small local JPEG batches to the existing counter.

## Build and use

1. Open this `android-phone-camera` folder in Android Studio and allow it to install the Android SDK components it requests.
2. Run it on an Android phone (Android 8+), grant Camera permission, and enter:
   - Vast URL: `http://VAST_IP:PUBLIC_8000_PORT`
   - API key: the same `CLOUD_INFERENCE_API_KEY` configured on Vast.
3. Mount the phone, use the rear camera, frame the road in the preview, and tap **Start sending**. The app saves the crop first, then publishes to `rtmp://VAST_IP:COUNTER_RTMP_PORT/phone` automatically. It defaults to Vast's existing `40073 → 10100` port mapping.

For a new installation, the fields are automatically filled from the repository's ignored parent [`.env`](../.env) file when you build the APK: `COUNTER_PUBLIC_URL` and `CLOUD_INFERENCE_API_KEY`. After starting once, the app stores them encrypted on the phone, so you do not need to re-enter them on later launches. Rebuild the APK after changing either value.

The app continues through a foreground service; Android displays an ongoing notification while it is active. It has no audio permission or audio stream.

## Vast setup

The Vast instance needs MediaMTX and the bridge running after the normal counter server:

```bash
cd /workspace/car-counter
set -a; source .env; set +a
/venv/main/bin/pip install -r requirements-stream.txt
nohup ./mediamtx mediamtx.yml > /root/mediamtx.log 2>&1 &
nohup /venv/main/bin/python -u stream_to_counter.py > /root/stream-bridge.log 2>&1 &
```

`./mediamtx` is the Linux MediaMTX release binary. Download it once from the official MediaMTX releases page and make it executable. The bridge waits until the phone sends its crop setup; this is expected before the first Start.

## Security

This initial version permits HTTP only because the current Vast server exposes plain HTTP. That exposes the crop request and API key on the network. RTMP itself uses the same key for authentication but is not encrypted. Use it only on a trusted network while testing; switch both endpoints to TLS before real deployment.

The build-time defaults are embedded in the APK, so never distribute this APK or commit a real `.env` file. They are a convenience for your own phone only.
