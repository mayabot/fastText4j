package com.mayabot.mynlp.fasttext.matrix

import com.mayabot.mynlp.fasttext.*
import java.io.File
import java.nio.ByteBuffer
import java.nio.channels.FileChannel
import java.util.*
import kotlin.math.min

/**
 * 不可变的Float矩阵
 */
interface FloatMatrix {
    fun rows(): Int
    fun cols(): Int
    operator fun get(row: Int): Vector
    operator fun get(i: Int, j: Int): Float

    fun write(channel: FileChannel)

    companion object {
        fun byteBufferMatrix(rows: Int, cols: Int) = MutableByteBufferMatrix(rows, cols, false)
        fun directByteBufferMatrix(rows: Int, cols: Int) = MutableByteBufferMatrix(rows, cols, true)
        fun floatArrayMatrix(rows: Int, cols: Int) = MutableFloatArrayMatrix(rows, cols)
        fun readOnlyFloatArrayMatrix(rows: Int, cols: Int,data: FloatArray) = FloatArrayMatrix(rows, cols, data)


        fun loadMatrix(file: File, mmap:Boolean): FloatMatrix {
            fun pages(total: Long, size: Int): Int = ((total + size.toLong() - 1) / size.toLong()).toInt()
            return if (mmap) {
                file.inputStream().channel.use {
                    val rows = it.readInt()
                    val cols = it.readInt()

                    //一个区域可以容纳多少行
                    var areaRows = 0
                    while (areaRows * cols < 268435456) {
                        areaRows += 10
                    }

                    val fileSize = it.size()
                    val arrayBytes = fileSize - 8
                    val areaCount = pages(arrayBytes, 4 * areaRows * cols)
                    val areaBytes = areaRows * cols * 4
                    val lastBytes = arrayBytes % (areaRows * cols * 4)

                    val list = ArrayList<ByteBuffer>()
                    for (a in 0 until areaCount) {
                        val len = if (a == areaCount - 1) lastBytes else areaBytes.toLong()
                        list += it.map(FileChannel.MapMode.READ_ONLY, 8 + a.toLong() * areaBytes, len)
                    }
                    AreaByteBufferMatrix(rows, cols, list)
                }
            } else {
                val dataInput = file.openAutoDataInput()
                val rows = dataInput.readInt()
                val cols = dataInput.readInt()
                val floatArray = FloatArray(rows * cols)
                for (i in 0 until rows * cols) {
                    floatArray[i] = dataInput.readFloat()
                }
                FloatMatrix.readOnlyFloatArrayMatrix(rows, cols, floatArray)
            }

        }
    }
}

abstract class BaseMatrix(val rows: Int, val cols: Int) : FloatMatrix {

    override fun rows(): Int {
        return rows
    }

    override fun cols(): Int {
        return cols
    }

    override fun toString(): String {
        if (rows == 0) {
            return ""
        }

        val b = StringBuilder()

        b.append("-".repeat(cols * 12))
        b.append("\n")

        for (i in 0 until min(20, rows)) {
            val row = get(i)

            for (j in 0 until cols) {
                b.append(row[j]).append("\t")
            }

            b.append("\n")
        }

        if (rows > 20) {
            b.append("....more....")
        }
        b.append("\n")

        return b.toString()
    }
}

/**
 * 行存储的只读矩阵。内存实现
 */
class FloatArrayMatrix(rows: Int, cols: Int, val data: FloatArray) : BaseMatrix(rows, cols), FloatMatrix {

    override fun write(channel: FileChannel) {
        val byteBuffer = ByteBuffer.allocate(cols * 4)
        val asFloatBuffer = byteBuffer.asFloatBuffer()
        for (row in 0 until rows) {
            asFloatBuffer.clear()
            asFloatBuffer.put(data, row * cols, cols)

            byteBuffer.position(0)
            byteBuffer.limit(cols*4)
            channel.write(byteBuffer)
        }
    }

    //private var length = rows * cols

    private val rowView = (0 until rows).mapIndexed { index, _ -> FloatArrayVector(data, index * cols, cols) }.toTypedArray()

    /**
     * 行视图
     */
    override operator fun get(row: Int): Vector {
        return rowView[row]
    }

    override operator fun get(i: Int, j: Int): Float {
        return data[i * cols + j]
    }
}

/**
 * 底层是一个ByteBuffer。实现只读版本的FloatMatrix
 */
class ByteBufferMatrix(rows: Int, cols: Int, val data: ByteBuffer) : BaseMatrix(rows, cols), FloatMatrix {

    // private var length = rows * cols
    //private val rowView = (0 until rows).mapIndexed { index, _ -> VectorDefault(data, index * cols, cols) }.toTypedArray()

    private fun index(i: Int, j: Int): Int {
        return i * cols + j
    }

    /**
     * 行视图
     */
    override operator fun get(row: Int): Vector {
        return ByteBufferVector(data, row * cols, cols)
    }

    override operator fun get(i: Int, j: Int): Float {
        return data.getFloat(index(i, j) shl 2)
    }

    override fun write(channel: FileChannel) {
        data.position(0)
        data.limit(data.capacity())

        channel.write(data)
    }
}


/**
 * 特殊版本的只读二维矩阵。
 * 在使用内存映射读取文件时，Java规定每个ByteBuffer不能超过2G大小
 */
class AreaByteBufferMatrix(rows: Int, cols: Int, val data: List<ByteBuffer>) : BaseMatrix(rows, cols), FloatMatrix {

    // private var length = rows * cols
    //private val rowView = (0 until rows).mapIndexed { index, _ -> VectorDefault(data, index * cols, cols) }.toTypedArray()

    val areaRows = data[0].capacity()/4/cols

    private fun index(i: Int, j: Int): Int {
        return i * cols + j
    }

    /**
     * 行视图
     */
    override operator fun get(row: Int): Vector {
        val area = row/areaRows
        val areaOffeet = row%areaRows
        return ByteBufferVector(data[area], areaOffeet * cols, cols)
    }

    override operator fun get(i: Int, j: Int): Float {

        val area = i/areaRows
        val areaOffeet = i%areaRows

        return data[area].getFloat(index(areaOffeet,j) shl 2)
    }

    override fun write(channel: FileChannel) {
        for (x in data) {
            x.position(0)
            x.limit(x.capacity())
            channel.write(x)
        }
    }
}
