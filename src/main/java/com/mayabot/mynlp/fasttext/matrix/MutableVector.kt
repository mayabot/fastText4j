package com.mayabot.mynlp.fasttext.matrix

import com.mayabot.mynlp.fasttext.checkArgument
import java.nio.ByteBuffer


/**
 * 可变向量
 * @author jimichan
 */
interface MutableVector : Vector {

    fun fill(v: Number)

    fun fill(call: (Int) -> Float)

    operator fun set(index: Int, value: Float)

    operator fun plusAssign(v: Vector)

    operator fun plusAssign(x: Pair<Number, Vector>)

    operator fun minusAssign(v: Vector)

    operator fun minusAssign(x: Pair<Number, Vector>)

    operator fun timesAssign(scale: Number)

    operator fun divAssign(scale: Number)

    fun putAll(v: FloatArray)

    fun zero()

    /**
     * 赋值
     */
    operator fun invoke(v: Vector)

}



class MutableFloatArrayVector(data: FloatArray,
                              offset: Int, length: Int) : FloatArrayVector(data, offset, length), MutableVector {

    constructor(size: Int) : this(FloatArray(size), 0, size)

    override fun fill(v: Number) {
        val v = v.toFloat()
        for (i in offset until offset + length) {
            data[i] = v
        }
    }

    override fun fill(call: (Int) -> Float) {
        var j = 0
        for (i in offset until offset + length) {
            data[i] = call(j++)
        }
    }

    override fun zero() {
        fill(0)
    }

    override fun invoke(v: Vector) {
        checkArgument(this.length() == v.length())

        var j = offset
        for (i in 0 until v.length()) {
            data[j++] = v[i]
        }
    }

    override fun set(index: Int, value: Float) {
        data[index+offset] = value
    }

    override fun putAll(v: FloatArray) {
        checkArgument(length == v.size)
        var j = 0

        for (i in offset until (offset + length)) {
            data[i] = v[j++]
        }
    }

    override fun plusAssign(v: Vector) {
        var j = 0
        for (i in offset until offset + length) {
            data[i] = data[i] + v[j++]
        }
    }

    override fun plusAssign(v: Pair<Number, Vector>) {
        val scale = v.first.toFloat()
        val vector = v.second
        if (scale == 1.0f) {
            plusAssign(v)
        }else{
            var j = 0
            for (i in offset until offset + length) {
                data[i] = data[i] + vector[j++] * scale
            }
        }
    }

    override fun minusAssign(x: Vector) {
        var j = 0
        for (i in offset until offset + length) {
            data[i] = data[i] - x[j++]
        }
    }

    override fun minusAssign(x: Pair<Number, Vector>) {
        val scale = x.first.toFloat()
        val vector = x.second
        var j = 0
        for (i in offset until offset + length) {
            data[i] = data[i] - vector[j++] * scale
        }
    }

    override fun timesAssign(scale: Number) {
        val scale = scale.toFloat()
        for (i in offset until offset + length) {
            data[i] *= scale
        }
    }

    override fun divAssign(scale: Number) {
        val scale = scale.toFloat()
        for (i in offset until offset + length) {
            data[i] /= scale
        }
    }
}



class MutableByteBufferVector(data: ByteBuffer,
                              offset: Int, length: Int) : ByteBufferVector(data, offset, length), MutableVector {

    constructor(size: Int) : this(ByteBuffer.allocate(size shl 2), 0, size)

    override fun fill(v: Number) {
        val v = v.toFloat()
        for (i in offset * 4 until (offset + length) * 4 step 4) {
            data.putFloat(i, v)
        }
    }

    override fun fill(call: (Int) -> Float) {
        var j = 0
        for (i in offset shl 2 until (offset + length) * 4 step 4) {
            data.putFloat(call(j++))
        }
    }

    override fun zero() {
        fill(0)
    }

    override fun invoke(v: Vector) {
        checkArgument(this.length() == v.length())
        for (i in 0 until v.length()) {
            this[i] = v[i]
        }
    }

    override fun set(index: Int, value: Float) {
        data.putFloat( (index+offset) shl 2, value)
    }

    override fun putAll(v: FloatArray) {
        checkArgument(length == v.size)
        var j = 0
        for (i in (offset * 4 until (offset + length) * 4 step 4)) {
            data.putFloat(i, v[j++])
        }
    }


    override fun plusAssign(v: Vector) {
        var j = 0
        for (i in (offset * 4 until (offset + length) * 4 step 4)) {
            data.putFloat(i, data.getFloat(i) + v[j++])
        }
    }

    override fun plusAssign(v: Pair<Number, Vector>) {
        val scale = v.first.toFloat()
        val vector = v.second
        var j = 0
        for (i in (offset * 4 until (offset + length) * 4 step 4)) {
            data.putFloat(i, data.getFloat(i) + vector[j++] * scale)
        }
    }

    override fun minusAssign(x: Vector) {
        var j = 0
        for (i in (offset * 4 until (offset + length) * 4 step 4)) {
            data.putFloat(i, data.getFloat(i) - x[j++])
        }
    }

    override fun minusAssign(x: Pair<Number, Vector>) {
        val scale = x.first.toFloat()
        val vector = x.second
        var j = 0
        for (i in (offset * 4 until (offset + length) * 4 step 4)) {
            data.putFloat(i, data.getFloat(i) - vector[j++] * scale)
        }
    }

    override fun timesAssign(scale: Number) {
        val scale = scale.toFloat()
        for (i in (offset * 4 until (offset + length) * 4 step 4)) {
            data.putFloat(i, data.getFloat(i) * scale)
        }
    }

    override fun divAssign(scale: Number) {
        val scale = scale.toFloat()
        if (scale != 0f) {
            for (i in (offset * 4 until (offset + length) * 4 step 4)) {
                data.putFloat(i, data.getFloat(i) / scale)
            }
        }
    }
}

inline operator fun ByteBuffer.set(i: Int, v: Float) {
    this.putFloat(i shl 2, v)
}
