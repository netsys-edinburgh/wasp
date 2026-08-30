package com.trainingontheedge.androidwaspclient.utils

import com.fasterxml.jackson.module.kotlin.jacksonObjectMapper
import com.fasterxml.jackson.module.kotlin.readValue
import java.io.File
import kotlin.random.Random

// Data class representing each triple and its frequency.
data class TripleCount(val m: Int, val n: Int, val k: Int, val count: Int)

// A class that loads the distribution from JSON, precomputes cumulative weights,
// and provides a low-latency sample() method.
class DistributionSampler(distributionFile: String, context: android.content.Context) {
    private val distribution: List<TripleCount>
    private val cumulativeWeights: List<Int>
    private val totalWeight: Int

    init {
        // Load distribution from JSON file
        val mapper = jacksonObjectMapper()
        val assetManager = context.assets
        distribution = mapper.readValue(assetManager.open(distributionFile))

        // Precompute cumulative weights for fast binary search sampling.
        val cumWeights = mutableListOf<Int>()
        var sum = 0
        for (tc in distribution) {
            sum += tc.count
            cumWeights.add(sum)
        }
        cumulativeWeights = cumWeights
        totalWeight = sum
    }

    /**
     * Get top K most common triples from the distribution.
     * @param k the number of triples to return.
     *
     * @return a list of TripleCount representing the top K triples.
     */
    fun getTopK(k: Int): List<TripleCount> {
        return distribution.sortedByDescending { it.count }.take(k)
    }

    /**
     * Samples a triple from the distribution in O(log N) time.
     *
     * @return a TripleCount representing the sampled triple, or null if the distribution is empty.
     */
    fun sample(): TripleCount {
        if (totalWeight == 0) return TripleCount(0, 0, 0, 0)

        // Generate a random integer in the range [0, totalWeight)
        val randomValue = Random.nextInt(totalWeight)

        // Binary search: find the smallest index where cumulativeWeights[index] > randomValue.
        var low = 0
        var high = cumulativeWeights.size - 1
        while (low < high) {
            val mid = (low + high) / 2
            if (randomValue < cumulativeWeights[mid]) {
                high = mid
            } else {
                low = mid + 1
            }
        }
        return distribution[low]
    }
}