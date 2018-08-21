package com.mayabot.mynlp.fasttext.matrix

import com.mayabot.mynlp.fasttext.checkArgument
import java.nio.ByteBuffer


/**
 * 只读的Float向量
 * @author jimichan
 */
interface Vector {

    fun subVector(offset: Int, size: Int): Vector

    fun plusTo(v:Vector): Vector

    fun minusTo(v:Vector): Vector

    operator fun get(index: Int): Float

    fun length(): Int

    fun prod(v: Vector): Float

    operator fun times(v: Vector): Float

    fun norm2(): Float

    fun norm2Pow(): Float

    fun check()

    fun copy(): MutableVector

    fun access(call: (Int, Float) -> Unit)

    companion object {
        fun byteBufferVector(size: Int) = MutableByteBufferVector(ByteBuffer.allocate(size shl 2), 0, size)
        fun directByteBufferVector(size: Int) = MutableByteBufferVector(ByteBuffer.allocateDirect(size shl 2), 0, size)
        fun floatArrayVector(size: Int) = MutableFloatArrayVector(size)

        fun dot(a: Vector, b: Vector): Float {
            return a * b
        }

        fun cosine(a: Vector, b: Vector): Float {
            val normA = a * a
            val normB = b * b
            return if (normA == 0.0f || normB == 0.0f) {
                0.0f
            } else (a * b / Math.sqrt((normA * normB).toDouble())).toFloat()
        }
    }
}


/**
 * 只读版本的基于FloatArray的向量
 */
open class FloatArrayVector(@JvmField protected val data: FloatArray,
                            @JvmField protected val offset: Int,
                            @JvmField protected val length: Int) : Vector {

    constructor(data: FloatArray) : this(data, 0, data.size)

    override fun subVector(offset: Int, size: Int): Vector {
        val result = FloatArray(size)
        for (i in 0 until size){
            result[i] += this[i + offset]
        }
        return FloatArrayVector(result,0,size)
    }

    override fun plusTo(v: Vector): Vector {
        checkArgument(length == v.length())
        val result = FloatArray(length)
        for (i in 0 until length){
            result[i] = this[i] + v[i]
        }
        return FloatArrayVector(result,0,length)
    }

    override fun minusTo(v: Vector): Vector {
        checkArgument(length == v.length())
        val result = FloatArray(length)
        for (i in 0 until length){
            result[i] = this[i] - v[i]
        }
        return FloatArrayVector(result,0,length)
    }

    /**
     * index 0 until length
     */
    override fun get(index: Int): Float {
        return data[index+offset]
    }

    override fun length(): Int = length

    override fun times(v: Vector) = this.prod(v)

    override fun prod(v: Vector): Float {
        //checkArgument(this.length() == v.length())
        var result = 0f
        var j = 0
        for (i in offset until offset + length) {
            result += data[i] * v[j++]
        }
        return result
    }

    override fun access(call: (Int, Float) -> Unit) {
        var j = 0
        for (i in offset until (offset + length)) {
            call(j++, data[i])
        }
    }

    /**
     * 第二范数 || v ||
     */
    override fun norm2(): Float {
        var sum = 0.0f
        for (i in offset until (offset + length)) {
            val x = data[i]
            sum += x * x
        }
        return Math.sqrt(sum.toDouble()).toFloat()
    }

    override fun norm2Pow(): Float {
        var sum = 0.0f
        for (i in offset until (offset + length)) {
            val x = data[i]
            sum += x * x
        }
        return sum
    }

    override fun check() {
        for (i in offset until offset + length) {
            val f = data[i]
            checkArgument(!f.isNaN())
            checkArgument(!f.isInfinite())
        }
    }

    override fun toString(): String {
        if (length() == 0)
            return "[]"

        val b = StringBuilder()
        b.append('[')
        val iMax = length() - 1
        val end = offset + iMax
        var i = offset
        while (true) {
            b.append(data[i])
            if (i == end)
                return b.append(']').toString()
            b.append(", ")
            i++
        }
    }

    override fun copy(): MutableVector {
        val dest = FloatArray(length)
        System.arraycopy(this.data, offset, dest, 0, length)
        return MutableFloatArrayVector(dest, 0, length)
    }
}

/**
 * 基于ByteBuffer存储的Vector实现
 * @author jimichan
 */
open class ByteBufferVector(@JvmField protected val data: ByteBuffer,
                            @JvmField protected val offset: Int,
                            @JvmField protected val length: Int) : Vector {

    constructor(data: ByteBuffer) : this(data, 0, data.capacity() / 4)

    override fun plusTo(v: Vector): Vector {
        checkArgument(length == v.length())
        val result = FloatArray(length)
        for (i in 0 until length){
            result[i] = this[i] + v[i]
        }
        return FloatArrayVector(result,0,length)
    }

    override fun minusTo(v: Vector): Vector {
        checkArgument(length == v.length())
        val result = FloatArray(length)
        for (i in 0 until length){
            result[i] = this[i] - v[i]
        }
        return FloatArrayVector(result,0,length)
    }

    override fun subVector(offset: Int, size: Int): Vector {
        val result = FloatArray(size)
        for (i in 0 until size){
            result[i] += this[i + offset]
        }
        return FloatArrayVector(result,0,size)
    }


    /**
     * index 0 until length
     */
    override fun get(index: Int): Float {
        return data.getFloat((index + offset) shl 2)
    }

    override fun length(): Int = length

    override fun times(v: Vector) = this.prod(v)

    override fun prod(v: Vector): Float {
        //checkArgument(this.length() == v.length())
        var result = 0f
        var j = 0
        for (i in offset until offset + length) {
            result += data.getFloat(i shl 2) * v[j++]
        }
        return result
    }

    override fun access(call: (Int, Float) -> Unit) {
        var j = 0
        for (i in offset shl 2 until (offset + length) * 4 step 4) {
            call(j++, data.getFloat(i))
        }
    }

    /**
     * 第二范数 || v ||
     */
    override fun norm2(): Float {
        var sum = 0.0f
        for (i in offset shl 2 until (offset + length) * 4 step 4) {
            val x = data.getFloat(i)
            sum += x * x
        }
        return Math.sqrt(sum.toDouble()).toFloat()
    }

    override fun norm2Pow(): Float {
        var sum = 0.0f
        for (i in offset shl 2 until (offset + length) * 4 step 4) {
            val x = data.getFloat(i)
            sum += x * x
        }
        return sum
    }

    override fun check() {
        for (i in offset until offset + length) {
            val f = data.getFloat(i shl 2)
            checkArgument(!f.isNaN())
            checkArgument(!f.isInfinite())
        }
    }

    override fun toString(): String {
        if (length() == 0)
            return "[]"

        val b = StringBuilder()
        b.append('[')
        val iMax = length() - 1
        val end = offset + iMax
        var i = offset
        while (true) {
            b.append(data.getFloat(i shl 2))
            if (i == end)
                return b.append(']').toString()
            b.append(", ")
            i++
        }
    }

    override fun copy(): MutableVector {
        val dest = ByteBuffer.allocate(length shl 2)
        for (i in offset shl 2 until (offset + length) * 4) {
            dest.put(data.get(i))
        }
        return MutableByteBufferVector(dest, 0, length)
    }
}
