package com.example.objectdetection

import android.Manifest
import android.graphics.*
import android.os.Bundle
import android.os.SystemClock
import android.util.Log
import android.util.Size
import android.widget.FrameLayout
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.*
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.*
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.*
import java.io.ByteArrayOutputStream
import java.util.Locale
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Portrait-first:
 * - PreviewView full screen (9:16), scaleType=FILL_CENTER → video lấp đầy, có thể crop 2 bên
 * - Camera stream 16:9 (cap 1080p) để chất lượng đẹp
 * - Canvas overlay map bbox từ (srcW x srcH) → (viewW x viewH) theo đúng quy tắc FILL_CENTER
 */
class CameraYoloActivity : ComponentActivity() {

    private val TAG = "CAM_YOLO"
    private var previewViewRef: PreviewView? = null
    private var permissionGranted = false
    private var modelReady = false

    // YOLO detector (bạn đã có class này)
    private val detector = YoloDetector(
        accelerationMode = YoloDetector.AccelerationMode.AUTO
    )

    // Overlay state: giữ kết quả gần nhất
    private var overlayState by mutableStateOf(
        OverlayState(emptyList(), 1920, 1080, "Initializing…")
    )

    // Tránh chồng chéo inference
    private val busy = AtomicBoolean(false)

    private val cameraPermission = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        permissionGranted = granted
        if (!granted) {
            overlayState = overlayState.copy(info = "Camera permission denied")
        } else {
            // Load model background trước
            lifecycleScope.launch(Dispatchers.Default) {
                try {
                    val res = detector.loadModel(this@CameraYoloActivity, "fruits_float32.tflite")
                    Log.i(TAG, "Model loaded with ${res.modeUsed}, ${res.loadTimeMs}ms")
                    modelReady = true
                    withContext(Dispatchers.Main) { overlayState = overlayState.copy(info = "Model ready") }
                } catch (t: Throwable) {
                    Log.e(TAG, "loadModel error", t)
                    withContext(Dispatchers.Main) { overlayState = overlayState.copy(info = "Load error: ${t.message}") }
                }
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        cameraPermission.launch(Manifest.permission.CAMERA)

        setContent {
            MaterialTheme {
                // full screen portrait, chừa system bars
                Box(
                    Modifier
                        .fillMaxSize()
                        .padding(WindowInsets.systemBars.asPaddingValues())
                ) {
                    // PreviewView full-size, FILL_CENTER để lấp đầy (có thể crop)
                    AndroidView(
                        factory = { ctx ->
                            PreviewView(ctx).apply {
                                layoutParams = FrameLayout.LayoutParams(
                                    FrameLayout.LayoutParams.MATCH_PARENT,
                                    FrameLayout.LayoutParams.MATCH_PARENT
                                )
                                scaleType = PreviewView.ScaleType.FILL_CENTER
                                previewViewRef = this
                                viewTreeObserver.addOnGlobalLayoutListener {
                                    Log.i(TAG, "PreviewView size=${width}x${height}")
                                }
                            }
                        },
                        modifier = Modifier
                            .align(Alignment.Center)
                            .fillMaxSize()
                    )

                    // Canvas overlay: fillMaxSize để trùng khít PreviewView
                    DetectionOverlayFillCenter(
                        state = overlayState,
                        modifier = Modifier
                            .align(Alignment.Center)
                            .fillMaxSize()
                    )

                    // Info bar
                    val density = LocalDensity.current
                    Text(
                        text = overlayState.info,
                        modifier = Modifier
                            .align(Alignment.TopCenter)
                            .padding(8.dp)
                    )

                    // Khi previewView có + permission + modelReady → bind camera
                    LaunchedEffect(previewViewRef, permissionGranted, modelReady) {
                        val pv = previewViewRef
                        if (permissionGranted && modelReady && pv != null) {
                            bindCamera(pv)
                        }
                    }
                }
            }
        }
    }

    private fun bindCamera(previewView: PreviewView) {
        val providerFuture = ProcessCameraProvider.getInstance(this)
        providerFuture.addListener({
            val provider = providerFuture.get()

            // Stream 16:9 (ưu tiên 1080p) — đẹp & phổ biến
            val selector16x9Cap1080 = ResolutionSelector.Builder()
                .setAspectRatioStrategy(AspectRatioStrategy.RATIO_16_9_FALLBACK_AUTO_STRATEGY)
                .setResolutionStrategy(
                    ResolutionStrategy(
                        Size(1920, 1080),
                        ResolutionStrategy.FALLBACK_RULE_CLOSEST_LOWER_THEN_HIGHER
                    )
                )
                .build()

            val preview = Preview.Builder()
                .setResolutionSelector(selector16x9Cap1080)
                .build()
                .also { it.setSurfaceProvider(previewView.surfaceProvider) }

            val analysis = ImageAnalysis.Builder()
                .setResolutionSelector(selector16x9Cap1080)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()

            // Analyzer: copy NV21; run YOLO ở background; overlay hiển thị kết quả gần nhất
            analysis.setAnalyzer(ContextCompat.getMainExecutor(this)) { image ->
                if (busy.getAndSet(true)) { image.close(); return@setAnalyzer }

                val rotation = image.imageInfo.rotationDegrees
                // Kích thước sau xoay (toạ độ tự nhiên đứng dọc nếu rot=90)
                val srcW = if (rotation % 180 == 0) image.width else image.height
                val srcH = if (rotation % 180 == 0) image.height else image.width

                // Copy NV21 & close sớm
                val w = image.width
                val h = image.height
                val nv21 = imageToNV21(image)
                image.close()

                lifecycleScope.launch(Dispatchers.Default) {
                    val t0 = SystemClock.elapsedRealtime()
                    try {
                        val bmp = nv21ToBitmap(nv21, w, h)
                        val rotated = rotateBitmap(bmp, rotation)

                        val pred = detector.predict(rotated, scoreThreshold = 0.25f)
                        val t = pred.times

                        val boxes = pred.detections.map {
                            RectF(it.getLeft(), it.getTop(), it.getRight(), it.getBottom())
                        }

                        val info = String.format(
                            Locale.US,
                            "src=%dx%d | prep=%dms inf=%dms post=%dms total=%dms det=%d",
                            srcW, srcH, t.preprocessMs, t.inferenceMs, t.postprocessMs, t.totalMs, pred.detections.size
                        )

                        withContext(Dispatchers.Main) {
                            overlayState = OverlayState(boxes, srcW, srcH, info)
                        }

                        rotated.recycle()
                        if (bmp != rotated) bmp.recycle()
                    } catch (e: Throwable) {
                        Log.e(TAG, "analyze error", e)
                        withContext(Dispatchers.Main) { overlayState = overlayState.copy(info = "Error: ${e.message}") }
                    } finally {
                        busy.set(false)
                        val elapsed = SystemClock.elapsedRealtime() - t0
                        Log.d(TAG, "frame handled in ${elapsed}ms")
                    }
                }
            }

            try {
                provider.unbindAll()
                provider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis)
                Log.i(TAG, "Camera bound (16:9 stream, portrait UI fill)")
                overlayState = overlayState.copy(info = "Camera ready")
            } catch (t: Throwable) {
                Log.e(TAG, "bind error", t)
                overlayState = overlayState.copy(info = "Bind error: ${t.message}")
            }
        }, ContextCompat.getMainExecutor(this))
    }
}

/* ---------------------------- Overlay & helpers ---------------------------- */

data class OverlayState(
    val boxes: List<RectF>,
    val srcWidth: Int,   // kích thước frame sau xoay (ví dụ 1080x1920 khi rot=90)
    val srcHeight: Int,
    val info: String
)

/**
 * Vẽ bbox theo quy tắc FILL_CENTER:
 * - scale = max(viewW/srcW, viewH/srcH)
 * - ảnh sau scale được căn giữa -> có offsetX/offsetY
 * → dùng cùng quy tắc với PreviewView.ScaleType.FILL_CENTER để khớp 100%
 */
@Composable
fun DetectionOverlayFillCenter(
    state: OverlayState,
    modifier: Modifier = Modifier
) {
    val strokePx = 3f
    Canvas(modifier = modifier) {
        if (state.srcWidth <= 0 || state.srcHeight <= 0) return@Canvas

        val viewW = size.width
        val viewH = size.height
        val sx = viewW / state.srcWidth
        val sy = viewH / state.srcHeight
        val scale = maxOf(sx, sy) // FILL_CENTER

        val drawnW = state.srcWidth * scale
        val drawnH = state.srcHeight * scale
        val offsetX = (viewW - drawnW) / 2f
        val offsetY = (viewH - drawnH) / 2f

        state.boxes.forEachIndexed { idx, r ->
            val hue = (idx * 37) % 360
            val paintColor = android.graphics.Color.HSVToColor(200, floatArrayOf(hue.toFloat(), 0.9f, 1f))
            val left = offsetX + r.left * scale
            val top  = offsetY + r.top  * scale
            val right = offsetX + r.right * scale
            val bottom = offsetY + r.bottom * scale

            drawRect(
                color = androidx.compose.ui.graphics.Color(paintColor),
                topLeft = Offset(left, top),
                size = androidx.compose.ui.geometry.Size(right - left, bottom - top),
                style = Stroke(width = strokePx)
            )
        }
    }
}

/** ImageProxy (YUV_420_888) -> NV21 byte[] (đơn giản, dễ hiểu) */
private fun imageToNV21(image: ImageProxy): ByteArray {
    val yPlane = image.planes[0].buffer
    val uPlane = image.planes[1].buffer
    val vPlane = image.planes[2].buffer

    val ySize = yPlane.remaining()
    val uSize = uPlane.remaining()
    val vSize = vPlane.remaining()
    val nv21 = ByteArray(ySize + uSize + vSize)

    // Y
    yPlane.get(nv21, 0, ySize)

    // Interleave VU (NV21)
    val w = image.width
    val h = image.height
    val uvHeight = h / 2

    val uRowStride = image.planes[1].rowStride
    val vRowStride = image.planes[2].rowStride
    val uPixelStride = image.planes[1].pixelStride
    val vPixelStride = image.planes[2].pixelStride

    var offset = ySize
    val uBuffer = image.planes[1].buffer
    val vBuffer = image.planes[2].buffer
    for (row in 0 until uvHeight) {
        var uIndex = row * uRowStride
        var vIndex = row * vRowStride
        for (col in 0 until w / 2) {
            nv21[offset++] = vBuffer.get(vIndex)
            nv21[offset++] = uBuffer.get(uIndex)
            uIndex += uPixelStride
            vIndex += vPixelStride
        }
    }
    return nv21
}

/** NV21 -> Bitmap qua JPEG (90) — thuận tiện để demo (không phải nhanh nhất). */
private fun nv21ToBitmap(nv21: ByteArray, width: Int, height: Int): Bitmap {
    val yuv = YuvImage(nv21, ImageFormat.NV21, width, height, null)
    val out = ByteArrayOutputStream()
    yuv.compressToJpeg(Rect(0, 0, width, height), 90, out)
    val bytes = out.toByteArray()
    return BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
}

/** Xoay bitmap theo rotationDegrees của CameraX (0/90/180/270). */
private fun rotateBitmap(src: Bitmap, rotationDegrees: Int): Bitmap {
    if (rotationDegrees % 360 == 0) return src
    val m = Matrix().apply { postRotate(rotationDegrees.toFloat()) }
    val bmp = Bitmap.createBitmap(src, 0, 0, src.width, src.height, m, true)
    if (bmp != src) src.recycle()
    return bmp
}
