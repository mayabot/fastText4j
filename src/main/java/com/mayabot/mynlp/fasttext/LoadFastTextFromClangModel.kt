package com.mayabot.mynlp.fasttext

import com.mayabot.blas.AutoDataInput
import com.mayabot.blas.FloatMatrix
import java.io.DataInputStream
import java.io.File
import java.io.IOException
import java.io.InputStream
import java.nio.ByteOrder

/**
 * 从C语言版本的FastText产生的模型文件
 */
object LoadFastTextFromClangModel {

    /**
     * Load binary model file. 这个二进制版本是C语言版本的模型
     * @param input C语言版本的模型的InputStream
     * @return FastTextModel
     * @throws Exception
     */
    @Throws(Exception::class)
    fun loadCModel(input: InputStream): FastText {
        input.buffered(1024*1024).use {
            val buffer = AutoDataInput(DataInputStream(it), ByteOrder.LITTLE_ENDIAN)

            //check model
            val magic = buffer.readInt()
            val version = buffer.readInt()

            if (magic != FASTTEXT_FILEFORMAT_MAGIC_INT32) {
                throw RuntimeException("Model file has wrong file format!")
            }

            if (version > FASTTEXT_VERSION) {
                throw RuntimeException("Model file has wrong file format! version is $version")
            }

            //Args
            val args_ = Args()
            args_.loadClang(buffer)

            if (version == 11 && args_.model == ModelName.sup) {
                // backward compatibility: old supervised models do not use char ngrams.
                args_.maxn = 0
            }

            //dictionary
            val dictionary = Dictionary(args_)
            dictionary.load(buffer)

            var input: FloatMatrix = FloatMatrix.floatArrayMatrix(0, 0)
            var qinput: QMatrix? = null

            val quantInput = buffer.readUnsignedByte() != 0
            if (quantInput) {
                qinput = QMatrix.load(buffer)
            } else {
                input = buffer.loadFloatMatrix()
            }

            if (!quantInput && dictionary.isPruned()) {
                throw RuntimeException("Invalid model file.\n"
                        + "Please download the updated model from www.fasttext.cc.\n"
                        + "See issue #332 on Github for more information.\n")
            }

            var output: FloatMatrix = FloatMatrix.floatArrayMatrix(0, 0)
            var qoutput: QMatrix? = null

            val qout = buffer.readUnsignedByte().toInt() != 0

            if (quantInput && qout) {
                qoutput = QMatrix.load(buffer)
            } else {
                output = buffer.loadFloatMatrix()
            }

            val model = Model(input, output, args_, 0)
            if (quantInput) {
                model.setQuantizePointer(qinput, qoutput)
            }

            if (args_.model == ModelName.sup) {
                model.setTargetCounts(dictionary.getCounts(EntryType.label))
            } else {
                model.setTargetCounts(dictionary.getCounts(EntryType.word))
            }

            return FastText(args_, dictionary,  model)
        }
    }

    /**
     * Load binary model file. 这个二进制版本是C语言版本的模型
     * @param modelPath
     * @return FastTextModel
     * @throws Exception
     */
    @Throws(Exception::class)
    fun loadCModel(modelFile: File): FastText {

        if (!(modelFile.exists() && modelFile.isFile && modelFile.canRead())) {
            throw IOException("Model file cannot be opened for loading!")
        }

        return loadCModel(modelFile.inputStream())
    }

    /**
     * Load binary model file. 这个二进制版本是C语言版本的模型
     * @param modelPath
     * @return FastTextModel
     * @throws Exception
     */
    @Throws(Exception::class)
    fun loadCModel(modelPath: String): FastText {
        val modelFile = File(modelPath)

        if (!(modelFile.exists() && modelFile.isFile && modelFile.canRead())) {
            throw IOException("Model file cannot be opened for loading!")
        }

        return loadCModel(modelFile.inputStream())
    }
}
