package com.example.objectdetection

import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.SystemClock
import org.tensorflow.lite.gpu.CompatibilityList
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import android.util.Log
import java.nio.ByteBuffer
import java.nio.ByteOrder

class YoloDetector(
    val inputSize: Int = 640,
    val channels: Int = 3,
    val bytesPerChannel: Int = 4,
    val batchSize: Int = 1,
    val outputSize: Int = 8400,
    val outputChannels: Int = 44,
    val nmsThreshold: Float = 0.45f,
    val accelerationMode: AccelerationMode = AccelerationMode.AUTO
) {

    companion object {
        private const val TAG = "YoloDetector"
    }

    /**
     * Data class để lưu thông tin detection
     */
    data class Detection(
        val x: Float, val y: Float, val width: Float, val height: Float,
        val confidence: Float, val classId: Int
    ) {
        fun getLeft(): Float = x - width / 2f
        fun getTop(): Float = y - height / 2f
        fun getRight(): Float = x + width / 2f
        fun getBottom(): Float = y + height / 2f
    }

    // Enum chọn chế độ tăng tốc
    enum class AccelerationMode {
        CPU,
        GPU,
        NNAPI,
        AUTO // Thử delegate theo thứ tự ưu tiên
    }

    // Interpreter và GPU delegate (sẽ được khởi tạo sau)
    var interpreter: Interpreter? = null
    var gpuDelegate: GpuDelegate? = null

    // Kết quả load
    data class LoadResult(
        val modeUsed: AccelerationMode,
        val loadTimeMs: Long
    )

    // Thời gian từng bước
    data class StepTimes(
        val preprocessMs: Long,
        val inferenceMs: Long,
        val postprocessMs: Long,
        val totalMs: Long
    )

    // Kết quả predict
    data class PredictResult(
        val detections: List<Detection>,
        val times: StepTimes
    )

    fun loadModel(
        context: Context,
        assetModelPath: String,
        mode: AccelerationMode = this.accelerationMode,
        cpuNumThreads: Int = Runtime.getRuntime().availableProcessors().coerceAtMost(4)
    ): LoadResult {
        // Dọn tài nguyên cũ (nếu có)
        interpreter?.close()
        gpuDelegate?.close()
        gpuDelegate = null

        val modelBuffer = loadModelFromAssets(context, assetModelPath)

        // Helper: build Interpreter từ options và đo thời gian
        fun build(options: Interpreter.Options, usedMode: AccelerationMode): LoadResult {
            val t0 = SystemClock.elapsedRealtimeNanos()
            interpreter = Interpreter(modelBuffer, options)
            val t1 = SystemClock.elapsedRealtimeNanos()
            return LoadResult(usedMode, (t1 - t0) / 1_000_000) // ms
        }

        // CPU options
        fun cpuOptions() = Interpreter.Options().apply {
            setNumThreads(cpuNumThreads)
            Log.i(TAG, "Max CPUs: ${Runtime.getRuntime().availableProcessors()}")
        }

        // GPU options (theo hướng dẫn bạn đưa)
        fun gpuOptions(): Interpreter.Options? {
            val compatList = CompatibilityList()
            if (compatList.isDelegateSupportedOnThisDevice) {
                val delegateOptions = compatList.bestOptionsForThisDevice
                gpuDelegate = GpuDelegate(delegateOptions)
                return Interpreter.Options().apply {
                    // Tránh crash trên một số GPU khi đọc output
                    setAllowBufferHandleOutput(false)
                    addDelegate(gpuDelegate)
                }
            }
            return null
        }

        // NNAPI options
        fun nnapiOptions(): Interpreter.Options {
            return Interpreter.Options().apply {
                @Suppress("DEPRECATION")
                setUseNNAPI(true)
                // Hoặc: addDelegate(NnApiDelegate()) nếu muốn kiểm soát chi tiết hơn
            }
        }

        return when (mode) {
            AccelerationMode.CPU -> {
                build(cpuOptions(), AccelerationMode.CPU)
            }

            AccelerationMode.GPU -> {
                val gpu = gpuOptions()
                    ?: return build(cpuOptions(), AccelerationMode.CPU)
                try {
                    build(gpu, AccelerationMode.GPU)
                } catch (_: Throwable) {  // Đổi từ "t" thành "_"
                    gpuDelegate?.close()
                    gpuDelegate = null
                    build(cpuOptions(), AccelerationMode.CPU)
                }
            }

            AccelerationMode.NNAPI -> {
                try {
                    build(nnapiOptions(), AccelerationMode.NNAPI)
                } catch (_: Throwable) {
                    // NNAPI lỗi → fallback CPU
                    build(cpuOptions(), AccelerationMode.CPU)
                }
            }

            AccelerationMode.AUTO -> {
                // Thứ tự ưu tiên: GPU → NNAPI → CPU
                // GPU
                gpuOptions()?.let { gpuOpt ->
                    try {
                        return build(gpuOpt, AccelerationMode.GPU)
                    } catch (_: Throwable) {
                        gpuDelegate?.close()
                        gpuDelegate = null
                    }
                }
                // NNAPI
                try {
                    return build(nnapiOptions(), AccelerationMode.NNAPI)
                } catch (_: Throwable) {
                    // ignore
                }
                // CPU
                build(cpuOptions(), AccelerationMode.CPU)
            }
        }
    }

    // ===== Helper: map model từ assets =====
    private fun loadModelFromAssets(context: Context, assetPath: String): MappedByteBuffer {
        val afd = context.assets.openFd(assetPath)
        FileInputStream(afd.fileDescriptor).channel.use { fc ->
            return fc.map(FileChannel.MapMode.READ_ONLY, afd.startOffset, afd.length)
        }
    }

    /**
     * Tiền xử lý ảnh từ Bitmap sang tensor input cho YOLO model
     * @param bitmap: Ảnh đầu vào dạng Bitmap
     * @return ByteBuffer chứa tensor đã được xử lý với shape [1,640,640,3]
     */
    fun preprocess(bitmap: Bitmap): ByteBuffer {

        // Bước 1: Resize bitmap về 640x640 (stretch - không giữ tỷ lệ)
        val resizedBitmap = resizeBitmapStretch(bitmap, inputSize, inputSize)

        // Bước 2: Tạo ByteBuffer với capacity phù hợp
        val inputBuffer = ByteBuffer.allocateDirect(
            batchSize * inputSize * inputSize * channels * bytesPerChannel
        )
        inputBuffer.order(ByteOrder.nativeOrder()) // Thường là LITTLE_ENDIAN

        // Bước 3: Chuyển đổi pixel values sang tensor format
        convertBitmapToTensor(resizedBitmap, inputBuffer)

        // Bước 4: Reset buffer position về đầu để đọc
        inputBuffer.rewind()

        // Clean up memory
        if (resizedBitmap != bitmap) {
            resizedBitmap.recycle()
        }

        return inputBuffer
    }

    /**
     * Resize bitmap về kích thước mới bằng cách stretch (làm méo)
     * Phù hợp với cách training model không dùng padding
     */
    private fun resizeBitmapStretch(bitmap: Bitmap, newWidth: Int, newHeight: Int): Bitmap {
        // Kiểm tra nếu bitmap đã bị recycle
        if (bitmap.isRecycled) {
            throw IllegalArgumentException("Cannot resize a recycled bitmap")
        }

        // Nếu size đã đúng, return copy để tránh modify bitmap gốc
        if (bitmap.width == newWidth && bitmap.height == newHeight) {
            return bitmap.copy(bitmap.config ?: Bitmap.Config.ARGB_8888, false)
        }

        val scaleWidth = newWidth.toFloat() / bitmap.width
        val scaleHeight = newHeight.toFloat() / bitmap.height

        val matrix = Matrix().apply {
            postScale(scaleWidth, scaleHeight)
        }

        return Bitmap.createBitmap(
            bitmap, 0, 0,
            bitmap.width, bitmap.height,
            matrix, true
        )
    }

    /**
     * Chuyển đổi bitmap thành tensor format trong ByteBuffer
     * Format: [batch, height, width, channels] = [1, 640, 640, 3]
     */
    private fun convertBitmapToTensor(bitmap: Bitmap, buffer: ByteBuffer) {

        // Tạo array để lưu pixel values
        val pixels = IntArray(bitmap.width * bitmap.height)
        bitmap.getPixels(pixels, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)

        // Duyệt qua từng pixel theo thứ tự: height -> width -> channels
        for (y in 0 until bitmap.height) {
            for (x in 0 until bitmap.width) {
                val pixelIndex = y * bitmap.width + x
                val pixel = pixels[pixelIndex]

                // Trích xuất RGB values từ pixel (ARGB format)
                val red = (pixel shr 16) and 0xFF
                val green = (pixel shr 8) and 0xFF
                val blue = pixel and 0xFF

                // Normalize về range [0.0, 1.0] và put vào buffer
                buffer.putFloat(red / 255.0f)   // Red channel
                buffer.putFloat(green / 255.0f) // Green channel
                buffer.putFloat(blue / 255.0f)  // Blue channel
            }
        }
    }


    /**
     * Post-processing: Chuyển output tensor thành danh sách detections
     * @param output: Output tensor từ model [1,44,8400]
     * @param originalWidth: Chiều rộng ảnh gốc (để scale coordinates)
     * @param originalHeight: Chiều cao ảnh gốc (để scale coordinates)
     * @param scoreThreshold: Ngưỡng confidence tối thiểu để giữ lại detection
     * @return List of Detection objects
     */
    fun postprocess(
        output: Array<Array<FloatArray>>,
        originalWidth: Int,
        originalHeight: Int,
        scoreThreshold: Float = 0.25f
    ): List<Detection> {
        val predictions = output[0] // Shape: [44,8400]
        val validDetections = mutableListOf<Detection>()

        for (i in 0 until outputSize) { // 8400 detections
            val centerX = predictions[0][i] * inputSize
            val centerY = predictions[1][i] * inputSize
            val width = predictions[2][i] * inputSize
            val height = predictions[3][i] * inputSize

            // Find max class score
            var maxConfidence = Float.NEGATIVE_INFINITY
            var maxClassId = 0

            for (classId in 0 until 40) {
                val classScore = predictions[4 + classId][i]
                if (classScore > maxConfidence) {
                    maxConfidence = classScore
                    maxClassId = classId
                }
            }

            if (maxConfidence > scoreThreshold) {
                validDetections.add(
                    Detection(
                        x = centerX,
                        y = centerY,
                        width = width,
                        height = height,
                        confidence = maxConfidence,
                        classId = maxClassId
                    )
                )
            }
        }

        // Apply NMS
        val sortedDetections = validDetections.sortedByDescending { it.confidence }
        val finalDetections = applyNMS(sortedDetections)

        // Scale coordinates to original image size
        val scaledDetections = scaleCoordinates(finalDetections, originalWidth, originalHeight)

        // Limit max detections
        return scaledDetections.take(50)
    }


    /**
     * Apply Non-Maximum Suppression để loại bỏ duplicate detections
     */
    private fun applyNMS(detections: List<Detection>): List<Detection> {
        if (detections.isEmpty()) return emptyList()

        // Sort by confidence (descending)
        val sortedDetections = detections.sortedByDescending { it.confidence }
        val finalDetections = mutableListOf<Detection>()

        for (detection in sortedDetections) {
            var shouldKeep = true

            // Check overlap với các detections đã chọn
            for (finalDetection in finalDetections) {
                val iou = calculateIoU(detection, finalDetection)
                if (iou > nmsThreshold) {
                    shouldKeep = false
                    break
                }
            }

            if (shouldKeep) {
                finalDetections.add(detection)
            }
        }
        return finalDetections
    }

    /**
     * Tính Intersection over Union (IoU) giữa 2 bounding boxes
     */
    private fun calculateIoU(
        det1: Detection,
        det2: Detection
    ): Float {
        val left = maxOf(det1.getLeft(), det2.getLeft())
        val top = maxOf(det1.getTop(), det2.getTop())
        val right = minOf(det1.getRight(), det2.getRight())
        val bottom = minOf(det1.getBottom(), det2.getBottom())

        if (left >= right || top >= bottom) return 0f

        val intersection = (right - left) * (bottom - top)
        val area1 = det1.width * det1.height
        val area2 = det2.width * det2.height
        val union = area1 + area2 - intersection

        return if (union > 0f) intersection / union else 0f
    }

    /**
     * Scale coordinates từ 640x640 về kích thước ảnh gốc
     */
    private fun scaleCoordinates(
        detections: List<Detection>,
        originalWidth: Int,
        originalHeight: Int
    ): List<Detection> {
        val scaleX = originalWidth.toFloat() / inputSize
        val scaleY = originalHeight.toFloat() / inputSize

        return detections.mapIndexed { index, detection ->
            val scaledDetection = detection.copy(
                x = detection.x * scaleX,
                y = detection.y * scaleY,
                width = detection.width * scaleX,
                height = detection.height * scaleY
            )
            scaledDetection
        }
    }


    /**
     * Inference: nhận ByteBuffer đầu vào và trả về output tensor [1, 44, 8400]
     * Chỉ làm nhiệm vụ chạy model (không đo thời gian ở đây để giữ hàm "sạch").
     */
    fun inference(inputBuffer: ByteBuffer): Array<Array<FloatArray>> {
        val tflite = interpreter ?: throw IllegalStateException("Interpreter is not initialized. Call loadModel() first.")

        // Tạo vùng nhớ cho output: [1, 44, 8400]
        val output = Array(batchSize) { Array(outputChannels) { FloatArray(outputSize) } }

        // Đảm bảo vị trí đọc buffer ở đầu
        inputBuffer.rewind()
        tflite.run(inputBuffer, output)
        return output
    }


    /**
     * Predict: thực hiện đủ 3 bước preprocess -> inference -> postprocess
     * Trả về danh sách detection và thời gian từng bước (ms).
     */
    fun predict(
        bitmap: Bitmap,
        scoreThreshold: Float = 0.25f
    ): PredictResult {
        val originalW = bitmap.width
        val originalH = bitmap.height

        // --- Preprocess ---
        val t0 = SystemClock.elapsedRealtimeNanos()
        val inputBuffer = preprocess(bitmap)
        val t1 = SystemClock.elapsedRealtimeNanos()

        // --- Inference ---
        val t2 = SystemClock.elapsedRealtimeNanos()
        val output = inference(inputBuffer)
        val t3 = SystemClock.elapsedRealtimeNanos()

        // --- Postprocess ---
        val t4 = SystemClock.elapsedRealtimeNanos()
        val detections = postprocess(
            output = output,
            originalWidth = originalW,
            originalHeight = originalH,
            scoreThreshold = scoreThreshold
        )
        val t5 = SystemClock.elapsedRealtimeNanos()

        val preprocessMs  = (t1 - t0) / 1_000_000
        val inferenceMs   = (t3 - t2) / 1_000_000
        val postprocessMs = (t5 - t4) / 1_000_000
        val totalMs       = (t5 - t0) / 1_000_000

        return PredictResult(
            detections = detections,
            times = StepTimes(
                preprocessMs = preprocessMs,
                inferenceMs = inferenceMs,
                postprocessMs = postprocessMs,
                totalMs = totalMs
            )
        )
    }

}