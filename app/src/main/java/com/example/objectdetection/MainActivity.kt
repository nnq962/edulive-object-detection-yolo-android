package com.example.objectdetection

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.tooling.preview.Preview
import com.example.objectdetection.ui.theme.ObjectDetectionTheme
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Log


class MainActivity : ComponentActivity() {

    private val detector = YoloDetector(accelerationMode = YoloDetector.AccelerationMode.NNAPI)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

        // Load model ở background (IO/Default)
        lifecycleScope.launch(Dispatchers.Default) {
            try {
                // 1) Load model
                val result = detector.loadModel(this@MainActivity, "fruits_float32.tflite")
                Log.i("YOLO", "Loaded with ${result.modeUsed}, time = ${result.loadTimeMs} ms")

                // 2) Đọc ảnh từ assets
                val bitmap = loadBitmapFromAssets("IMG_20250812_111140.jpg")

                // 3) Chạy predict
                val pred = detector.predict(bitmap, scoreThreshold = 0.25f)

                // Log thời gian từng bước
                val t = pred.times
                Log.i(
                    "YOLO", "Preprocess: ${t.preprocessMs} ms, " +
                            "Inference: ${t.inferenceMs} ms, " +
                            "Postprocess: ${t.postprocessMs} ms, " +
                            "Total: ${t.totalMs} ms"
                )

                // Log một vài detection (nếu có)
                pred.detections.take(5).forEachIndexed { i, det ->
                    Log.i(
                        "YOLO",
                        "Det#$i class=${det.classId} conf=${"%.2f".format(det.confidence)} " +
                                "box=(${det.getLeft().toInt()}, ${
                                    det.getTop().toInt()
                                }, ${det.getRight().toInt()}, ${det.getBottom().toInt()})"
                    )
                }
            } catch (t: Throwable) {
                Log.e("YOLO", "Error during detection", t)
            }
        }

        setContent {
            ObjectDetectionTheme {
                Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
                    Greeting(
                        name = "Android",
                        modifier = Modifier.padding(innerPadding)
                    )
                }
            }
        }
    }

    private fun loadBitmapFromAssets(path: String): Bitmap {
        assets.open(path).use { input ->
            return BitmapFactory.decodeStream(input)
                ?: error("Cannot decode bitmap from assets: $path")
        }
    }

}

@Composable
fun Greeting(name: String, modifier: Modifier = Modifier) {
    Text(
        text = "Hello $name!",
        modifier = modifier
    )
}

@Preview(showBackground = true)
@Composable
fun GreetingPreview() {
    ObjectDetectionTheme {
        Greeting("Android")
    }
}