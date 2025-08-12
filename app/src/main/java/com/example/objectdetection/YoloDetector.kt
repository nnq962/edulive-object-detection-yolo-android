// Top 1 android Edulive
package com.example.objectdetection

import android.content.Context
import android.graphics.Bitmap
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import android.graphics.Matrix
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import kotlin.math.exp
import java.io.Closeable

class YoloDetector : Closeable {
    companion object {
        private const val INPUT_SIZE = 640
        private const val CHANNELS = 3
        private const val BYTES_PER_CHANNEL = 4
        private const val BATCH_SIZE = 1
        private const val OUTPUT_SIZE = 8400
        private const val OUTPUT_CHANNELS = 44

        private val CLASS_NAMES = arrayOf(
            "apple", "apricot", "avocado", "banana", "blueberry", "breadfruit",
            "cantaloupe", "cherry", "coconut", "cranberry", "custard-apple",
            "dragon-fruit", "durian", "grape", "guava", "kiwi", "lemon", "lime",
            "longan", "lychee", "mango", "mangosteen", "mulberry", "orange",
            "papaya", "passion-fruit", "peach", "pear", "persimmon", "pineapple",
            "plum", "pomegranate", "pomelo", "rambutan", "sapodilla", "starfruit",
            "strawberry", "tamarind", "watermelon", "winter-melon"
        )
    }

    /**
     * Data class để lưu thông tin detection
     */
    data class Detection(
        val x: Float, val y: Float, val width: Float, val height: Float,
        val confidence: Float, val classId: Int, val className: String
    ) {
        fun getLeft(): Float = x - width / 2f
        fun getTop(): Float = y - height / 2f
        fun getRight(): Float = x + width / 2f
        fun getBottom(): Float = y + height / 2f
    }

    private var interpreter: Interpreter? = null
    private var gpuDelegate: GpuDelegate? = null
    private var hexagonDelegate: Any? = null // Hexagon delegate
    private val confidenceThreshold = 0.5f
    private val nmsThreshold = 0.45f
    private fun sigmoid(x: Float): Float = 1f / (1f + exp(-x))


    // Add acceleration mode enum
    enum class AccelerationMode {
        CPU_ONLY,
        GPU,
        NNAPI,
        HEXAGON,
        AUTO // Try delegates in order of preference
    }

    // Hàm này add thêm vào để chạy 1 lần nên đặt là predict
    fun detect(bitmap: Bitmap, rotationDeg: Int, scoreThreshold: Float): Pair<List<Detection>, Long> {
        val rotated = if (rotationDeg % 360 == 0) bitmap else {
            val m = Matrix().apply { postRotate(rotationDeg.toFloat()) }
            Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, m, true)
        }
        val input = preprocessImage(rotated)
        val t0 = System.currentTimeMillis()
        val output = runInference(input)
        val dt = System.currentTimeMillis() - t0
        val dets = postprocessOutput(output, rotated.width, rotated.height, scoreThreshold)
        return dets to dt
    }

    /**
     * Post-processing: Chuyển output tensor thành danh sách detections
     * @param output: Output tensor từ model [1,44,8400]
     * @param originalWidth: Chiều rộng ảnh gốc (để scale coordinates)
     * @param originalHeight: Chiều cao ảnh gốc (để scale coordinates)
     * @param scoreThreshold: Ngưỡng confidence tối thiểu để giữ lại detection
     * @return List of Detection objects
     */
    fun postprocessOutput(
        output: Array<Array<FloatArray>>,
        originalWidth: Int,
        originalHeight: Int,
        scoreThreshold: Float = 0.25f
    ): List<Detection> {
        val predictions = output[0] // Shape: [44,8400]
        val validDetections = mutableListOf<Detection>()

        for (i in 0 until OUTPUT_SIZE) { // 8400 detections
            val centerX = predictions[0][i] * INPUT_SIZE
            val centerY = predictions[1][i] * INPUT_SIZE
            val width = predictions[2][i] * INPUT_SIZE
            val height = predictions[3][i] * INPUT_SIZE

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
                        classId = maxClassId,
                        className = CLASS_NAMES[maxClassId]
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

        android.util.Log.i("YOLO", "🧹 NMS: ${detections.size} → ${finalDetections.size} detections")
        return finalDetections
    }

    /**
     * Tính Intersection over Union (IoU) giữa 2 bounding boxes
     */
    private fun calculateIoU(det1: Detection, det2: Detection): Float {
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
        val scaleX = originalWidth.toFloat() / INPUT_SIZE
        val scaleY = originalHeight.toFloat() / INPUT_SIZE

        // DEBUG: In scale factors
        android.util.Log.i("YOLO", "🔍 Scale factors: scaleX=$scaleX, scaleY=$scaleY")

        return detections.mapIndexed { index, detection ->
            val scaledDetection = detection.copy(
                x = detection.x * scaleX,
                y = detection.y * scaleY,
                width = detection.width * scaleX,
                height = detection.height * scaleY
            )

            // DEBUG: Log coordinates cho detection đầu tiên
            if (index == 0 && AppConfig.ENABLE_DEBUG_LOG) {
                android.util.Log.i("YOLO", "🔍 Detection coordinate mapping:")
                android.util.Log.i("YOLO", "  Original image: ${originalWidth}x${originalHeight}")
                android.util.Log.i("YOLO", "  Scale factors: scaleX=$scaleX, scaleY=$scaleY")
                android.util.Log.i("YOLO", "  Raw detection (640x640): center=(${detection.x}, ${detection.y}), size=(${detection.width}, ${detection.height})")
                android.util.Log.i("YOLO", "  Scaled to original: center=(${scaledDetection.x}, ${scaledDetection.y}), size=(${scaledDetection.width}, ${scaledDetection.height})")
                android.util.Log.i("YOLO", "  Bounding box: left=${scaledDetection.getLeft()}, top=${scaledDetection.getTop()}, right=${scaledDetection.getRight()}, bottom=${scaledDetection.getBottom()}")
            }

            scaledDetection
        }
    }

    /**
     * Tiền xử lý ảnh từ Bitmap sang tensor input cho YOLO model
     * @param bitmap: Ảnh đầu vào dạng Bitmap
     * @return ByteBuffer chứa tensor đã được xử lý với shape [1,640,640,3]
     */
    fun preprocessImage(bitmap: Bitmap): ByteBuffer {

        // Bước 1: Resize bitmap về 640x640 (stretch - không giữ tỷ lệ)
        val resizedBitmap = resizeBitmapStretch(bitmap, INPUT_SIZE, INPUT_SIZE)

        // Bước 2: Tạo ByteBuffer với capacity phù hợp
        val inputBuffer = ByteBuffer.allocateDirect(
            BATCH_SIZE * INPUT_SIZE * INPUT_SIZE * CHANNELS * BYTES_PER_CHANNEL
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
     * Load model with acceleration
     */
    fun loadModel(
        context: Context, 
        modelPath: String, 
        numThreads: Int = 2,
        delegate: Int = DELEGATE_CPU
    ) {
        try {
            // Clean up any existing resources
            close()
            
            val modelBuffer = loadModelFile(context, modelPath)
            val options = Interpreter.Options().apply {
                setNumThreads(numThreads.coerceIn(1, 4)) // Ensure valid thread count
            }

            setupAcceleration(options, delegate)
            
            // Create interpreter
            interpreter = Interpreter(modelBuffer, options)
            android.util.Log.i("YOLO", "🚀 Model loaded successfully!")
            
            logModelInfo()

        } catch (e: Exception) {
            android.util.Log.e("YOLO", "❌ Error loading model: ${e.message}")
            // Clean up on failure
            close()
            throw e
        }
    }



    private fun setupAcceleration(options: Interpreter.Options, delegate: Int): Boolean {
        return when (delegate) {
            DELEGATE_CPU -> {
                android.util.Log.i("YOLO", "🖥️ Using CPU only")
                options.setUseNNAPI(false)
                options.setNumThreads(4) // Use 4 threads for CPU
                true
            }
            DELEGATE_GPU -> {
                val success = setupGpuDelegateSafe(options)
                if (!success) {
                    android.util.Log.w("YOLO", "🔄 GPU failed, using CPU with 4 threads")
                    options.setUseNNAPI(false)
                    options.setNumThreads(4) // Fallback to 4 threads
                }
                success
            }
            DELEGATE_NNAPI -> {
                val success = setupNnapiDelegate(options)
                if (!success) {
                    android.util.Log.w("YOLO", "🔄 NNAPI failed, using CPU with 4 threads")
                    options.setUseNNAPI(false)
                    options.setNumThreads(4) // Fallback to 4 threads
                }
                success
            }
            else -> {
                android.util.Log.i("YOLO", "🖥️ Unknown delegate, using CPU")
                options.setUseNNAPI(false)
                options.setNumThreads(4)
                true
            }
        }
    }

    private fun setupGpuDelegateSafe(options: Interpreter.Options): Boolean {
        return try {
            android.util.Log.i("YOLO", "🔍 Attempting GPU delegate creation...")

            val compatList = CompatibilityList()
            android.util.Log.i("YOLO", "🔍 GPU compatibility check: ${compatList.isDelegateSupportedOnThisDevice}")

            if (compatList.isDelegateSupportedOnThisDevice) {
                // Clean up any existing GPU delegate first
                gpuDelegate?.close()
                gpuDelegate = null
                
                // Use EXACTLY the same approach that works on your device
                val delegateOptions = compatList.bestOptionsForThisDevice
                android.util.Log.i("YOLO", "🔍 Creating GPU delegate with bestOptionsForThisDevice...")
                gpuDelegate = GpuDelegate(delegateOptions)
                
                // Add the delegate to options
                options.addDelegate(gpuDelegate)
                android.util.Log.i("YOLO", "✅ GPU Delegate created successfully with bestOptions!")
                true
            } else {
                android.util.Log.w("YOLO", "⚠️ GPU not supported on this device, will use CPU with 4 threads")
                options.setNumThreads(4)
                false
            }
        } catch (e: Exception) {
            android.util.Log.e("YOLO", "❌ GPU Delegate failed: ${e.message}")
            android.util.Log.e("YOLO", "❌ Exception: ${e.javaClass.simpleName}")
            e.printStackTrace()
            
            // Clean up failed delegate
            try {
                gpuDelegate?.close()
            } catch (closeException: Exception) {
                android.util.Log.w("YOLO", "Failed to close GPU delegate: ${closeException.message}")
            }
            gpuDelegate = null
            
            // Fallback to CPU with 4 threads
            options.setNumThreads(4)
            false
        }
    }

    private fun setupNnapiDelegate(options: Interpreter.Options): Boolean {
        return try {
            options.setUseNNAPI(true)
            android.util.Log.i("YOLO", "✅ NNAPI Delegate enabled")
            true
        } catch (e: Exception) {
            android.util.Log.w("YOLO", "⚠️ NNAPI failed: ${e.message}")
            options.setUseNNAPI(false)
            false
        }
    }


    /**
     * Chạy inference với input tensor
     * @param inputBuffer: Tensor đầu vào đã preprocessing
     * @return Output tensor với shape [1,44,8400]
     */
    fun runInference(inputBuffer: ByteBuffer): Array<Array<FloatArray>> {
        val interpreter = this.interpreter
            ?: throw IllegalStateException("Model chưa được load! Gọi loadModel() trước.")

        // Tạo output buffer với shape [1,44,8400]
        val output = Array(BATCH_SIZE) {
            Array(OUTPUT_CHANNELS) {
                FloatArray(OUTPUT_SIZE)
            }
        }

        try {
            // Reset input buffer position
            inputBuffer.rewind()

            // Validate input buffer size
            val expectedSize = BATCH_SIZE * INPUT_SIZE * INPUT_SIZE * CHANNELS * BYTES_PER_CHANNEL
            if (inputBuffer.capacity() != expectedSize) {
                throw IllegalArgumentException("Input buffer size mismatch. Expected: $expectedSize, Got: ${inputBuffer.capacity()}")
            }

            // Đo thời gian inference
            val startTime = System.currentTimeMillis()

            // Chạy model với try-catch để bắt GPU crashes
            interpreter.run(inputBuffer, output)

            val inferenceTime = System.currentTimeMillis() - startTime
            android.util.Log.d("YOLO", "⚡ Inference time: ${inferenceTime}ms")

            return output

        } catch (e: Exception) {
            android.util.Log.e("YOLO", "❌ Error during inference: ${e.message}")
            android.util.Log.e("YOLO", "❌ Exception type: ${e.javaClass.simpleName}")
            
            // If it's a native crash or GPU-related error, we should recreate the interpreter
            if (e.message?.contains("native") == true || 
                e.message?.contains("GPU") == true || 
                e is RuntimeException) {
                android.util.Log.w("YOLO", "⚠️ Possible GPU delegate issue detected, consider fallback to CPU")
            }
            
            throw e
        }
    }

    /**
     * Load model file từ assets
     */
    private fun loadModelFile(context: Context, modelPath: String): MappedByteBuffer {
        val fileDescriptor = context.assets.openFd(modelPath)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    private fun isGpuSupported(): Boolean {
        return try {
            val compatibilityList = CompatibilityList()
            compatibilityList.isDelegateSupportedOnThisDevice
        } catch (e: Exception) {
            android.util.Log.w("YOLO", "Cannot check GPU support: ${e.message}")
            false
        }
    }

    /**
     * Log thông tin model (for debugging)
     */
    private fun logModelInfo() {
        val interpreter = this.interpreter ?: return

        try {
            android.util.Log.i("YOLO", "📊 Model Info:")
            android.util.Log.i("YOLO", "  Input count: ${interpreter.inputTensorCount}")
            android.util.Log.i("YOLO", "  Output count: ${interpreter.outputTensorCount}")

            // Input tensor info
            val inputTensor = interpreter.getInputTensor(0)
            android.util.Log.i("YOLO", "  Input shape: ${inputTensor.shape().contentToString()}")
            android.util.Log.i("YOLO", "  Input type: ${inputTensor.dataType()}")

            // Output tensor info
            val outputTensor = interpreter.getOutputTensor(0)
            android.util.Log.i("YOLO", "  Output shape: ${outputTensor.shape().contentToString()}")
            android.util.Log.i("YOLO", "  Output type: ${outputTensor.dataType()}")

        } catch (e: Exception) {
            android.util.Log.w("YOLO", "Cannot get model info: ${e.message}")
        }
    }

    /**
     * Kiểm tra shape của tensor đã tạo (for debugging)
     */
    fun getTensorShape(): IntArray {
        return intArrayOf(BATCH_SIZE, INPUT_SIZE, INPUT_SIZE, CHANNELS)
    }

    /**
     * Tính kích thước ByteBuffer cần thiết
     */
    fun getInputBufferSize(): Int {
        return BATCH_SIZE * INPUT_SIZE * INPUT_SIZE * CHANNELS * BYTES_PER_CHANNEL
    }


    /**
     * Debug function để test các acceleration modes
     */
    fun debugAccelerationSupport(context: Context) {
        android.util.Log.i("YOLO", "🔍 === TESTING ACCELERATION SUPPORT ===")

        // Test GPU Compatibility Check
        android.util.Log.i("YOLO", "📱 Testing GPU compatibility...")
        try {
            val compatibilityList = CompatibilityList()
            val isGpuSupported = compatibilityList.isDelegateSupportedOnThisDevice
            android.util.Log.i("YOLO", "  GPU compatibility: $isGpuSupported")

            if (isGpuSupported) {
                val bestOptions = compatibilityList.bestOptionsForThisDevice
                android.util.Log.i("YOLO", "  Best GPU options available: $bestOptions")
            }

        } catch (e: Exception) {
            android.util.Log.e("YOLO", "  ❌ GPU compatibility test failed: ${e.message}")
        }

        // Test NNAPI
        android.util.Log.i("YOLO", "🧠 Testing NNAPI support...")
        try {
            val testOptions = Interpreter.Options()
            testOptions.setUseNNAPI(true)
            android.util.Log.i("YOLO", "  ✅ NNAPI seems available")
        } catch (e: Exception) {
            android.util.Log.e("YOLO", "  ❌ NNAPI test failed: ${e.message}")
        }

        // Test CPU
        android.util.Log.i("YOLO", "🖥️ CPU always available")

        android.util.Log.i("YOLO", "🏁 === ACCELERATION TEST COMPLETED ===")
        android.util.Log.i("YOLO", "💡 Recommendation: Try AUTO mode to fallback gracefully")
    }

    /**
     * Clean up resources including delegates
     */
    override fun close() {
        interpreter?.close()
        gpuDelegate?.close()

        hexagonDelegate?.let { delegate ->
            try {
                val closeMethod = delegate.javaClass.getMethod("close")
                closeMethod.invoke(delegate)
            } catch (e: Exception) {
                android.util.Log.w("YOLO", "Failed to close Hexagon delegate: ${e.message}")
            }
        }

        interpreter = null
        gpuDelegate = null
        hexagonDelegate = null
        android.util.Log.i("YOLO", "🧹 YoloDetector closed")
    }
}