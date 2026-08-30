package com.example.edgeclientlitert.runner

import android.content.Context
import android.util.Log
import com.trainingontheedge.androidwaspclient.utils.DistributionSampler
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.nio.ByteBuffer
import kotlin.collections.LinkedHashMap
import kotlin.system.measureNanoTime

/**
 * Wrapper for GPURunner that stores the tensors
 */
class RunnerCacheEntry<T: Runner>(private val runner: T) {
    fun matmul(matrixA: FloatArray, matrixB: FloatArray, m: Int, k: Int, n: Int): FloatArray {
        val tensorA = TensorBuffer.createFixedSize(intArrayOf(1, m, k), DataType.FLOAT32)
        val tensorB = TensorBuffer.createFixedSize(intArrayOf(1, k, n), DataType.FLOAT32)
        val tensorC = TensorBuffer.createFixedSize(intArrayOf(1, m, n), DataType.FLOAT32)
        tensorA.loadArray(matrixA)
        tensorB.loadArray(matrixB)
        val outputs = mapOf(0 to tensorC.buffer.clear())
        runner.getRunner().runForMultipleInputsOutputs(arrayOf(tensorA.buffer, tensorB.buffer), outputs)
        return tensorC.floatArray
    }

    fun matmul(matrixA: ByteBuffer, matrixB: ByteBuffer, m: Int, k: Int, n: Int): FloatArray {
        val tensorA = TensorBuffer.createFixedSize(intArrayOf(1, m, k), DataType.FLOAT32)
        val tensorB = TensorBuffer.createFixedSize(intArrayOf(1, k, n), DataType.FLOAT32)
        val tensorC = TensorBuffer.createFixedSize(intArrayOf(1, m, n), DataType.FLOAT32)
        tensorA.loadBuffer(matrixA)
        tensorB.loadBuffer(matrixB)
        val outputs = mapOf(0 to tensorC.buffer.clear())
        runner.getRunner().runForMultipleInputsOutputs(arrayOf(tensorA.buffer, tensorB.buffer), outputs)
        return tensorC.floatArray
    }

    fun matmulTimed(matrixA: FloatArray, matrixB: FloatArray, m: Int, k: Int, n: Int): Pair<FloatArray, Pair<Long, Long>> {
        val tensorA: TensorBuffer
        val tensorB: TensorBuffer
        val tensorC: TensorBuffer

        val creationTime = measureNanoTime {
            tensorA = TensorBuffer.createFixedSize(intArrayOf(1, m, k), DataType.FLOAT32)
            tensorB = TensorBuffer.createFixedSize(intArrayOf(1, k, n), DataType.FLOAT32)
            tensorC = TensorBuffer.createFixedSize(intArrayOf(1, m, n), DataType.FLOAT32)
            tensorA.loadArray(matrixA)
            tensorB.loadArray(matrixB)
        }
        val executionTime = measureNanoTime {
            val outputs = mapOf(0 to tensorC.buffer.clear())
            runner.getRunner().runForMultipleInputsOutputs(arrayOf(tensorA.buffer, tensorB.buffer), outputs)
        }
        return tensorC.floatArray to Pair(creationTime, executionTime)
    }

    fun getRunner(): T {
        return runner
    }

    fun close() {
        runner.close()
    }
}

/**
 * Data class representing a key for a Runner instance.
 * m, k, n represent the matrix dimensions used for resizing the input tensors.
 */
data class RunnerKey(val m: Int, val k: Int, val n: Int)

/**
 * A simple LRU cache for Runner instances.
 * On a cache miss, instead of creating a new Runner, the least recently used runner is reconfigured.
 *
 * @param context the Android context.
 * @param maxSize the maximum number of Runner instances to cache.
 * @param prepopulate if true, prepopulate the cache with default keys using DistributionSampler.
 * @param runnerFactory a lambda that creates a new Runner given a RunnerKey.
 */
class RunnerCache<T : Runner>(
    private val context: Context,
    private val maxSize: Int,
    prepopulate: Boolean,
    private val runnerFactory: (RunnerKey) -> T
) {
    private val cache = object : LinkedHashMap<RunnerKey, RunnerCacheEntry<T>>(maxSize, 0.75f, true) {
        override fun removeEldestEntry(eldest: MutableMap.MutableEntry<RunnerKey, RunnerCacheEntry<T>>?): Boolean {
            // We won't remove automatically since we always reuse an existing entry on miss.
            return false
        }
    }

    // Sampler to get default keys from a distribution file.
    private val sampler = DistributionSampler("distribution.json", context)

    init {
        if (prepopulate) {
            // Get top K default keys from the sampler.
            val defaultTriplets = sampler.getTopK(maxSize)
            for (triplet in defaultTriplets) {
                val key = RunnerKey(triplet.m, triplet.k, triplet.n)
                val runner = runnerFactory(key).apply {
                    loadModel(context, "mul.tflite")
                    resizeInput(0, intArrayOf(1, key.m, key.k))
                    resizeInput(1, intArrayOf(1, key.k, key.n))
                    allocateTensors()
                }
                cache[key] = RunnerCacheEntry(runner)
            }
        }
    }

    /**
     * getTiming: Given two float arrays and dimensions m, k, n,
     * returns the result of matmulTimed using a runner that is configured for these dimensions.
     *
     * If the key is already in the cache, it is returned.
     * Otherwise, the least recently used runner is reconfigured to match the new dimensions.
     */
    @Synchronized
    fun getTiming(
        matrixA: FloatArray,
        matrixB: FloatArray,
        m: Int, k: Int, n: Int
    ): Pair<FloatArray, Pair<Long, Long>> {
        val key = RunnerKey(m, k, n)
        val runner: RunnerCacheEntry<T> = if (cache.containsKey(key)) {
            cache[key]!!
        } else {
            if (cache.isNotEmpty()) {
//                Log.d("RunnerCache", "Cache miss for key $key")
                // Get the least-recently-used runner (eldest entry).
                val eldestEntry = cache.entries.iterator().next()
                val reconfigRunner = cache.remove(eldestEntry.key)!!
                // Reconfigure this runner with new dimensions.
                reconfigRunner.getRunner().resizeInput(0, intArrayOf(1, m, k))
                reconfigRunner.getRunner().resizeInput(1, intArrayOf(1, k, n))
                reconfigRunner.getRunner().allocateTensors()
                reconfigRunner
            } else {
                // No runner in cache, so create a new one.
                val newRunner = runnerFactory(key).apply {
                    loadModel(context, "mul.tflite")
                    resizeInput(0, intArrayOf(1, m, k))
                    resizeInput(1, intArrayOf(1, k, n))
                    allocateTensors()
                }
                RunnerCacheEntry(newRunner)

            }.also { cache[key] = it }
        }
//        Log.d("RunnerCache", "Using key $key for dimensions $m x $k x $n")
        return runner.matmulTimed(matrixA, matrixB, m, k, n)
    }

    /**
     * get: Convenience method that returns just the result (without timing)
     */
    @Synchronized
    fun get(
        matrixA: FloatArray,
        matrixB: FloatArray,
        m: Int, k: Int, n: Int
    ): FloatArray {
        val key = RunnerKey(m, k, n)
        val runner: RunnerCacheEntry<T> = if (cache.containsKey(key)) {
            cache[key]!!
        } else {
            if (cache.isNotEmpty()) {
                // Get the least-recently-used runner (eldest entry).
                val eldestEntry = cache.entries.iterator().next()
                val reconfigRunner = cache.remove(eldestEntry.key)!!
                // Reconfigure this runner with new dimensions.
                reconfigRunner.getRunner().resizeInput(0, intArrayOf(1, m, k))
                reconfigRunner.getRunner().resizeInput(1, intArrayOf(1, k, n))
                reconfigRunner.getRunner().allocateTensors()
                reconfigRunner
            } else {
                // No runner in cache, so create a new one.
                val newRunner = runnerFactory(key).apply {
                    loadModel(context, "mul.tflite")
                    resizeInput(0, intArrayOf(1, m, k))
                    resizeInput(1, intArrayOf(1, k, n))
                    allocateTensors()
                }
                RunnerCacheEntry(newRunner)

            }.also { cache[key] = it }
        }
        return runner.matmul(matrixA, matrixB, m, k, n)
    }

    /**
     * get: Convenience method that returns just the result (without timing)
     */
    @Synchronized
    fun get(
        matrixA: ByteBuffer,
        matrixB: ByteBuffer,
        m: Int, k: Int, n: Int
    ): FloatArray {
        val key = RunnerKey(m, k, n)
        val runner: RunnerCacheEntry<T> = if (cache.containsKey(key)) {
            cache[key]!!
        } else {
            if (cache.isNotEmpty()) {
                // Get the least-recently-used runner (eldest entry).
                val eldestEntry = cache.entries.iterator().next()
                val reconfigRunner = cache.remove(eldestEntry.key)!!
                // Reconfigure this runner with new dimensions.
                reconfigRunner.getRunner().resizeInput(0, intArrayOf(1, m, k))
                reconfigRunner.getRunner().resizeInput(1, intArrayOf(1, k, n))
                reconfigRunner.getRunner().allocateTensors()
                reconfigRunner
            } else {
                // No runner in cache, so create a new one.
                val newRunner = runnerFactory(key).apply {
                    loadModel(context, "mul.tflite")
                    resizeInput(0, intArrayOf(1, m, k))
                    resizeInput(1, intArrayOf(1, k, n))
                    allocateTensors()
                }
                RunnerCacheEntry(newRunner)

            }.also { cache[key] = it }
        }
        return runner.matmul(matrixA, matrixB, m, k, n)
    }

    @Synchronized
    fun clear() {
        cache.values.forEach { it.close() }
        cache.clear()
    }
}