package com.mayabot.mynlp.fasttext

import com.carrotsearch.hppc.IntArrayList
import java.util.*


/**
 * 训练模型和计算模型都需要一个setTargetCounts方法。
 * BaseModel主要目的是构建negative sampling或者hierarchical softmax
 * @author jimichan
 * @see Model
 * @see TrainModel
 */
open class BaseModel(
        @JvmField val args_: Args,
        randomSeed: Number,
        @JvmField var outputMatrixSize: Int) {

    // used for negative sampling:
    @JvmField
    protected var negatives: IntArray = IntArray(0)

    @JvmField
    protected var negpos: Int = 0

    // used for hierarchical softmax:
    @JvmField
    protected var paths: MutableList<IntArray> = ArrayList()
    @JvmField
    protected var codes: MutableList<BooleanArray> = ArrayList()
    @JvmField
    protected var tree: MutableList<Node> = ArrayList()


    @Transient
    @JvmField
    val rng: Random = Random(randomSeed.toLong())

    fun setTargetCounts(counts: LongArray) {
        checkArgument(counts.size == outputMatrixSize)
        if (args_.loss == LossName.ns) {
            initTableNegatives(counts)
        } else if (args_.loss == LossName.hs) {
            buildTree(counts)
        }
    }

    private fun initTableNegatives(counts: LongArray) {
        val negatives_ = IntArrayList(counts.size)

        var z = counts.map { sqrt(it) }.sum()
        val size = counts.size

        val xxn = NEGATIVE_TABLE_SIZE / z
        for (i in 0 until size) {
            val c = sqrt(counts[i])
            var j = 0
            while (j < c * xxn) {
                negatives_.add(i)
                j++
            }
        }
        negatives = negatives_.toArray()
        shuffle(negatives, rng)
    }

    private fun buildTree(counts: LongArray) {
        val pathsLocal = ArrayList<IntArray>(outputMatrixSize)
        val codesLocal = ArrayList<BooleanArray>(outputMatrixSize)
        val treeLocal = ArrayList<Node>(2 * outputMatrixSize - 1)

        for (i in 0 until 2 * outputMatrixSize - 1) {
            treeLocal.add(Node().apply {
                this.parent = -1
                this.left = -1
                this.right = -1
                this.count = 1000000000000000L// 1e15f;
                this.binary = false
            })
        }

        for (i in 0 until outputMatrixSize) {
            treeLocal[i].count = counts[i]
        }

        var leaf = outputMatrixSize - 1
        var node = outputMatrixSize
        for (i in outputMatrixSize until 2 * outputMatrixSize - 1) {
            val mini = IntArray(2)
            for (j in 0..1) {
                if (leaf >= 0 && treeLocal[leaf].count < treeLocal[node].count) {
                    mini[j] = leaf--
                } else {
                    mini[j] = node++
                }
            }
            treeLocal[i].apply {
                this.left = mini[0]
                this.right = mini[1]
                this.count = treeLocal[mini[0]].count + treeLocal[mini[1]].count
            }
            treeLocal[mini[0]].parent = i
            treeLocal[mini[1]].parent = i
            treeLocal[mini[1]].binary = true
        }

        for (i in 0 until outputMatrixSize) {
            val path = ArrayList<Int>()
            val code = ArrayList<Boolean>()

            var j = i
            while (treeLocal[j].parent != -1) {
                path.add(treeLocal[j].parent - outputMatrixSize)
                code.add(treeLocal[j].binary)
                j = treeLocal[j].parent
            }
            pathsLocal.add(path.toIntArray())
            codesLocal.add(code.toBooleanArray())
        }

        this.paths = pathsLocal
        this.codes = codesLocal
        this.tree = treeLocal
    }

    companion object {
        private val tSigmoid: FloatArray = FloatArray(SIGMOID_TABLE_SIZE + 1) { i ->
            val x = (i * 2 * MAX_SIGMOID).toFloat() / SIGMOID_TABLE_SIZE - MAX_SIGMOID
            (1.0f / (1.0f + Math.exp((-x).toDouble()))).toFloat()
        }

        private val tLog: FloatArray = FloatArray(LOG_TABLE_SIZE + 1) { i ->
            val x = (i.toFloat() + 1e-5f) / LOG_TABLE_SIZE
            Math.log(x.toDouble()).toFloat()
        }

        fun log(x: Float): Float {
            if (x > 1.0f) {
                return 0.0f
            }
            val i = (x * LOG_TABLE_SIZE).toInt()
            return tLog[i]
        }

        fun sigmoid(x: Float): Float {
            return when {
                x < -MAX_SIGMOID -> 0.0f
                x > MAX_SIGMOID -> 1.0f
                else -> {
                    val i = ((x + MAX_SIGMOID) * SIGMOID_TABLE_SIZE / MAX_SIGMOID.toFloat() / 2f).toInt()
                    tSigmoid[i]
                }
            }
        }
    }
}