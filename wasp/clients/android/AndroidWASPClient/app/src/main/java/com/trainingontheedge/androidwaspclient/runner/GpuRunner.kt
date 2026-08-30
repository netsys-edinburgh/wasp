package com.example.edgeclientlitert.runner

import android.content.Context
import android.util.Log
import org.tensorflow.lite.DataType
import org.tensorflow.lite.InterpreterApi
import org.tensorflow.lite.gpu.GpuDelegateFactory
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import kotlin.system.measureNanoTime

class GpuRunner: Runner() {

    override fun loadModel(context: Context, assetName: String): Boolean {
        try {
            val modelBuffer = FileUtil.loadMappedFile(context, "mul.tflite")
            val gpuOptions = GpuDelegateFactory.Options()
                .setPrecisionLossAllowed(false)
                .setInferencePreference(1) // Sustained speed
                .setForceBackend(GpuDelegateFactory.Options.GpuBackend.OPENCL)
            val gpuInterpreterOption = InterpreterApi.Options()
                .setRuntime(InterpreterApi.Options.TfLiteRuntime.FROM_APPLICATION_ONLY)
                .addDelegateFactory(GpuDelegateFactory(gpuOptions))
            interpreter = InterpreterApi.create(modelBuffer, gpuInterpreterOption)
            Log.d("Interpreter", "GPU interpreter initialized.")
        } catch (e: Exception) {
            Log.e("Interpreter", "Error loading model: $e")
            return false
        }
        return true
    }

    override fun matmul(matrixA: FloatArray, matrixB: FloatArray, An: Int, Am: Int, Bn: Int, Bm: Int, numThreads: Int): FloatArray {
        val tensorA: TensorBuffer = TensorBuffer.createFixedSize(intArrayOf(1, An, Am), DataType.FLOAT32)
        val tensorB: TensorBuffer = TensorBuffer.createFixedSize(intArrayOf(1, Bn, Bm), DataType.FLOAT32)
        val tensorC: TensorBuffer = TensorBuffer.createFixedSize(intArrayOf(1, An, Bm), DataType.FLOAT32)
        interpreter.resizeInput(0, intArrayOf(1, An, Am))
        interpreter.resizeInput(1, intArrayOf(1, Bn, Bm))
        interpreter.allocateTensors()
        tensorA.loadArray(matrixA)
        tensorA.buffer.clear()
        tensorB.loadArray(matrixB)
        tensorB.buffer.clear()
        val outputs = mapOf(0 to tensorC.buffer.clear())
        interpreter.runForMultipleInputsOutputs(arrayOf(tensorA.buffer, tensorB.buffer), outputs)
        return tensorC.floatArray
    }

    override fun matmulTimed(matrixA: FloatArray, matrixB: FloatArray, An: Int, Am: Int, Bn: Int, Bm: Int, numThreads: Int): Pair<FloatArray, Pair<Long, Long>> {
        val tensorA: TensorBuffer
        val tensorB: TensorBuffer
        val tensorC: TensorBuffer
        val outputs: Map<Int, Any>

        val creationTime = measureNanoTime {
            tensorA = TensorBuffer.createFixedSize(intArrayOf(1, An, Am), DataType.FLOAT32)
            tensorB = TensorBuffer.createFixedSize(intArrayOf(1, Bn, Bm), DataType.FLOAT32)
            tensorC = TensorBuffer.createFixedSize(intArrayOf(1, An, Bm), DataType.FLOAT32)
            interpreter.resizeInput(0, intArrayOf(1, An, Am))
            interpreter.resizeInput(1, intArrayOf(1, Bn, Bm))
            interpreter.allocateTensors()
            tensorA.loadArray(matrixA)
            tensorA.buffer.clear()
            tensorB.loadArray(matrixB)
            tensorB.buffer.clear()
            outputs = mapOf(0 to tensorC.buffer.clear())
        }
        interpreter.runForMultipleInputsOutputs(arrayOf(tensorA.buffer, tensorB.buffer), outputs)
        val time = interpreter.getLastNativeInferenceDurationNanoseconds()
        return Pair(tensorC.floatArray, Pair(creationTime, time))
    }

    override fun matmul(
        matrixA: TensorBuffer,
        matrixB: TensorBuffer,
        numThreads: Int
    ): TensorBuffer {
        val tensorC: TensorBuffer = TensorBuffer.createFixedSize(intArrayOf(1, matrixA.shape[1], matrixB.shape[2]), DataType.FLOAT32)
        interpreter.resizeInput(0, intArrayOf(1, matrixA.shape[1], matrixA.shape[2]))
        interpreter.resizeInput(1, intArrayOf(1, matrixB.shape[1], matrixB.shape[2]))
        interpreter.allocateTensors()
        val outputs = mapOf(0 to tensorC.buffer.clear())

        interpreter.runForMultipleInputsOutputs(arrayOf(matrixA.buffer, matrixB.buffer), outputs)

        return tensorC
    }

    override fun matmulTimed(
        matrixA: TensorBuffer,
        matrixB: TensorBuffer,
        numThreads: Int
    ): Pair<TensorBuffer, Pair<Long, Long>> {
        val tensorC: TensorBuffer
        val outputs: Map<Int, Any>

        val creationTime = measureNanoTime {
            tensorC = TensorBuffer.createFixedSize(intArrayOf(1, matrixA.shape[1], matrixB.shape[2]), DataType.FLOAT32)
            interpreter.resizeInput(0, intArrayOf(1, matrixA.shape[1], matrixA.shape[2]))
            interpreter.resizeInput(1, intArrayOf(1, matrixB.shape[1], matrixB.shape[2]))
            interpreter.allocateTensors()
            outputs = mapOf(0 to tensorC.buffer.clear())
        }
        interpreter.runForMultipleInputsOutputs(arrayOf(matrixA.buffer, matrixB.buffer), outputs)
        val time = interpreter.getLastNativeInferenceDurationNanoseconds()
        return Pair(tensorC, Pair(creationTime, time))
    }

    override fun resizeInput(index: Int, dims: IntArray) {
        interpreter.resizeInput(index, dims)
    }

    override fun allocateTensors() {
        interpreter.allocateTensors()
    }

    override fun close() {
        interpreter.close()
    }

    override fun getRunner(): InterpreterApi {
        return interpreter
    }

}