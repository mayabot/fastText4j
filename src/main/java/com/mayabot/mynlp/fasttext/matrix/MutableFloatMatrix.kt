package com.mayabot.mynlp.fasttext.matrix

import java.nio.ByteBuffer
import java.nio.channels.FileChannel
import java.util.*


/**
 * 可变的Matrix
 * @author jimichan
 */
interface MutableFloatMatrix : FloatMatrix {
    override operator fun get(row: Int): MutableVector
    operator fun set(i: Int, j: Int, v: Float)
    fun fill(v: Float)
    fun uniform(a: Number)
}

/**
 * 数据全部加载到内存的.
 * 按行存储的内存矩阵
 */
class MutableFloatArrayMatrix(rows: Int, cols: Int) : BaseMatrix(rows, cols), MutableFloatMatrix {

    override fun write(channel: FileChannel) {
        val byteBuffer = ByteBuffer.allocateDirect(cols * 4)
        val asFloatBuffer = byteBuffer.asFloatBuffer()

        for (row in 0 until rows) {
            asFloatBuffer.clear()
            asFloatBuffer.put(data, row * cols, cols)

            byteBuffer.position(0)
            byteBuffer.limit(cols*4)

            channel.write(byteBuffer)
        }
    }

    private var length = rows * cols

    var data = FloatArray(length)

    private val rnd: Random = Random()

    private val rowview = (0 until rows).mapIndexed { index, i -> MutableFloatArrayVector(data, index * cols, cols) }.toTypedArray()

    constructor() : this(0, 0)

    override fun uniform(a: Number) {
        var a = a.toFloat()
        val lower = -a
        for (i in 0 until length) {
            data[i] = rnd.nextFloat() * (a - lower) + lower
        }
    }

    /**
     * 均值为0
     * @param sd 标准差
     */
    fun gaussRandom(sd: Number) {
        var sd = sd.toFloat()
        for (i in 0 until length) {
            data[i] = (rnd.nextGaussian() * sd).toFloat()
        }
    }

    /**
     * 行视图
     */
    override operator fun get(row: Int): MutableVector {
        return rowview[row]
    }

    override operator fun get(i: Int, j: Int): Float {
        return data[i * cols + j]
    }

    override operator fun set(i: Int, j: Int, v: Float) {
        data[i * cols + j] = v
    }

    override fun fill(v: Float) {
        for (i in 0 until length) {
            data[i] = v
        }
    }

}



class MutableByteBufferMatrix(rows: Int, cols: Int, direct: Boolean = true) : BaseMatrix(rows, cols), MutableFloatMatrix {

    private var length = rows * cols

    val data = if (direct) ByteBuffer.allocateDirect(length shl 2)!! else ByteBuffer.allocateDirect(length shl 2)!!

    private val rnd: Random = Random()

    private val rowview = (0 until rows).mapIndexed { index, _ -> MutableByteBufferVector(data, index * cols, cols) }.toTypedArray()

    constructor() : this(0, 0)

    private fun index(i: Int, j: Int): Int {
        return i * cols + j
    }

    override fun uniform(a: Number) {
        var a = a.toFloat()
        val lower = -a
        for (i in 0 until length step 4) {
            data.putFloat(i, rnd.nextFloat() * (a - lower) + lower)
        }
    }

    /**
     * 均值为0
     * @param sd 标准差
     */
    fun gaussRandom(sd: Number) {
        var sd = sd.toFloat()
        for (i in 0 until length step 4) {
            data.putFloat(i, (rnd.nextGaussian() * sd).toFloat())
        }
    }

    /**
     * 行视图
     */
    override operator fun get(row: Int): MutableVector {
        return rowview[row]
    }

    /**
     * get cell
     */
    override operator fun get(i: Int, j: Int): Float {
        return data.getFloat(index(i, j) shl 2)
    }

    /**
     * set cell
     */
    override operator fun set(i: Int, j: Int, v: Float) {
        data.putFloat(index(i, j) shl 2, v)
    }


    override fun fill(v: Float) {
        for (i in 0 until length step 4) {
            data.putFloat(i, v)
        }
    }

    override fun write(channel: FileChannel) {
        data.position(0)
        data.limit(data.capacity())

        channel.write(data)
    }

}