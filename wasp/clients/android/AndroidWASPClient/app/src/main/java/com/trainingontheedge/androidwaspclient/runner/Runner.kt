package com.example.edgeclientlitert.runner

import android.content.Context
import org.tensorflow.lite.InterpreterApi
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer

abstract class Runner {
    protected lateinit var interpreter: InterpreterApi

    abstract fun loadModel(context: Context, assetName: String): Boolean
    abstract fun matmul(matrixA: FloatArray, matrixB: FloatArray, An: Int, Am: Int, Bn: Int, Bm: Int, numThreads: Int): FloatArray
    abstract fun matmul(matrixA: TensorBuffer, matrixB: TensorBuffer, numThreads: Int): TensorBuffer
    abstract fun matmulTimed(matrixA: FloatArray, matrixB: FloatArray, An: Int, Am: Int, Bn: Int, Bm: Int, numThreads: Int): Pair<FloatArray, Pair<Long, Long>>
    abstract fun matmulTimed(matrixA: TensorBuffer, matrixB: TensorBuffer, numThreads: Int): Pair<TensorBuffer, Pair<Long, Long>>
    abstract fun close()
    abstract fun resizeInput(index: Int, dims: IntArray)
    abstract fun allocateTensors()
    abstract fun getRunner(): InterpreterApi
}