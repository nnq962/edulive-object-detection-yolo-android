package com.example.objectdetection

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.fillMaxSize
//import androidx.compose.foundation.layout.padding
//import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.tooling.preview.Preview
import com.example.objectdetection.ui.theme.ObjectDetectionTheme
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.launch
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Log
import androidx.compose.foundation.Image
import androidx.compose.runtime.*
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.foundation.layout.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.Alignment
//import androidx.compose.material3.Text
//import java.io.File
//import java.io.FileOutputStream
import kotlinx.coroutines.Dispatchers
//import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext


class MainActivity : ComponentActivity() {

    private val detector = YoloDetector(accelerationMode = YoloDetector.AccelerationMode.GPU)

    val labels = listOf(
        "apple",
        "apricot",
        "avocado",
        "banana",
        "blueberry",
        "breadfruit",
        "cantaloupe",
        "cherry",
        "coconut",
        "cranberry",
        "custard-apple",
        "dragon-fruit",
        "durian",
        "grape",
        "guava",
        "kiwi",
        "lemon",
        "lime",
        "longan",
        "lychee",
        "mango",
        "mangosteen",
        "mulberry",
        "orange",
        "papaya",
        "passion-fruit",
        "peach",
        "pear",
        "persimmon",
        "pineapple",
        "plum",
        "pomegranate",
        "pomelo",
        "rambutan",
        "sapodilla",
        "starfruit",
        "strawberry",
        "tamarind",
        "watermelon",
        "winter-melon"
    )

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

        // State hiển thị ảnh kết quả
        var resultBitmap by mutableStateOf<Bitmap?>(null)
        var infoText by mutableStateOf("Loading...")

        // Load model ở background (IO/Default)
        lifecycleScope.launch(Dispatchers.Default) {
            try {
                // 1) Load model
                val result = detector.loadModel(this@MainActivity, "fruits_float16.tflite")
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

                // (Tuỳ chọn) danh sách nhãn nếu bạn có
                val labels: List<String>? = labels

                // Vẽ kết quả
                val vis = detector.drawDetectionsOnBitmap(bitmap, pred.detections, labels)

                withContext(Dispatchers.Main) {
                    resultBitmap = vis
                    infoText = "Detections: ${pred.detections.size} | " +
                            "prep=${t.preprocessMs}ms inf=${t.inferenceMs}ms post=${t.postprocessMs}ms"
                }

            } catch (t: Throwable) {
                Log.e("YOLO", "Error during detection", t)
            }
        }

        setContent {
            ObjectDetectionTheme {
                Box(Modifier.fillMaxSize()) {
                    resultBitmap?.let { bmp ->
                        Image(
                            bitmap = bmp.asImageBitmap(),
                            contentDescription = "Detections",
                            modifier = Modifier.fillMaxSize(),
                            contentScale = ContentScale.Fit
                        )
                    } ?: run {
                        // Chưa có ảnh => hiện text
                        Text(
                            text = infoText,
                            modifier = Modifier.align(Alignment.Center)
                        )
                    }
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