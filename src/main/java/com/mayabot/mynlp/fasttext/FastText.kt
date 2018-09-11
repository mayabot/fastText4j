package com.mayabot.mynlp.fasttext

import com.carrotsearch.hppc.IntArrayList
import com.google.common.base.Charsets
import com.google.common.base.Stopwatch
import com.google.common.collect.ImmutableList
import com.google.common.collect.Iterables
import com.google.common.collect.Lists
import com.google.common.collect.Sets
import com.google.common.io.Files
import com.google.common.primitives.Floats
import com.mayabot.blas.*
import com.mayabot.blas.Vector
import java.io.File
import java.io.IOException
import java.io.InputStream
import java.text.DecimalFormat
import java.util.*
import java.util.concurrent.TimeUnit
import kotlin.math.exp
import kotlin.system.exitProcess

const val FASTTEXT_VERSION = 12
const val FASTTEXT_FILEFORMAT_MAGIC_INT32 = 793712314

data class FloatIntPair(@JvmField var first: Float, @JvmField var second: Int)
data class FloatStringPair(@JvmField var first: Float, @JvmField var second: String){
    override fun toString(): String {
        return "[$second,$first]"
    }
}

class FastText(internal val args: Args,
               internal val dict: Dictionary,
               internal val model: Model
) {

    /**
     * 是否量化. 指的是隐藏层或者LEFT或者是词向量是否向量化
     */
    val quant = model.quant

    val input = model.input
    val output = model.output

    lateinit var wordVectors: FloatMatrix


    /**
     * 预测分类标签
     *
     * @param tokens
     * @param k
     * @return
     */
    fun predict(tokens: Iterable<String>, k: Int): List<FloatStringPair> {
        val tokens2 = Iterables.concat(tokens, listOf(EOS))
        val words = IntArrayList()
        val labels = IntArrayList()

        dict.getLine(tokens2, words, labels)

        if (words.isEmpty) {
            return ImmutableList.of()
        }
        val hidden = MutableByteBufferVector(args.dim)
        val output = MutableByteBufferVector(dict.nlabels())

        val modelPredictions = Lists.newArrayListWithCapacity<FloatIntPair>(k)

        model.predict(words, k, modelPredictions, hidden, output)

        return modelPredictions.map { x -> FloatStringPair(exp(x.first), dict.getLabel(x.second)) }
    }


    private fun findNN(wordVectors: FloatMatrix, queryVec: Vector, k: Int, sets: Set<String>): List<FloatStringPair> {

        var queryNorm = queryVec.norm2()
        if (Math.abs(queryNorm) < 1e-8) {
            queryNorm = 1f
        }

        val mostSimilar = (0 until k).map { FloatStringPair(-1f,"") }.toList().toTypedArray()
        val mastSimilarLast = mostSimilar.size - 1

        for (i in 0 until dict.nwords()) {
            val dp = wordVectors[i] *queryVec / queryNorm
            val last = mostSimilar[mastSimilarLast]
            if (dp > last.first) {
                last.first = dp
                last.second = dict.getWord(i)

                mostSimilar.sortByDescending { it.first }
            }
        }

        val result = Lists.newArrayList<FloatStringPair>()
        for (r in mostSimilar) {
            if (r.first != -1f && !sets.contains(r.second)) {
                result.add(r)
            }
        }

        return result
    }


    /**
     * NearestNeighbor
     */
    fun nearestNeighbor(wordQuery: String, k: Int): List<FloatStringPair> {
        if (!this::wordVectors.isInitialized) {
            val stopwatch = Stopwatch.createStarted()
            wordVectors = FloatMatrix.floatArrayMatrix(dict.nwords,args.dim).apply {
                preComputeWordVectors(this)
            }
            stopwatch.stop()
            println("Init wordVectors martix use time ${stopwatch.elapsed(TimeUnit.MILLISECONDS)} ms")
        }
        val queryVec = getWordVector(wordQuery)
        val sets = HashSet<String>()
        sets.add(wordQuery)
        return findNN(wordVectors, queryVec, k, sets)
    }

    /**
     * Query triplet (A - B + C)?
     * @param A
     * @param B
     * @param C
     * @param k
     */
    fun analogies(A: String, B: String, C: String, k: Int): List<FloatStringPair> {
        if (!this::wordVectors.isInitialized) {
            val stopwatch = Stopwatch.createStarted()
            wordVectors = FloatMatrix.floatArrayMatrix(dict.nwords,args.dim).apply {
                preComputeWordVectors(this)
            }
            stopwatch.stop()
            println("Init wordVectors martix use time ${stopwatch.elapsed(TimeUnit.MILLISECONDS)} ms")
        }

        val buffer = Vector.floatArrayVector(args.dim)
        val query = Vector.floatArrayVector(args.dim)

        getWordVector(buffer, A)
        query += buffer

        getWordVector(buffer, B)
        query += -1f to buffer

        getWordVector(buffer, C)
        query += buffer

        val sets = Sets.newHashSet(A, B, C)

        return findNN(wordVectors, query, k, sets)
    }


    /**
     * 计算所有词的向量。
     * 之所以向量都除以norm进行归一化。因为使用者。使用dot表达相似度，也会除以query vector的norm。然后归一化。
     * 最后距离结构都是0 ~ 1 的数字
     * @param wordVectors
     */
    private fun preComputeWordVectors(wordVectors: MutableFloatMatrix) {
        val vec = Vector.floatArrayVector(args.dim)
        wordVectors.fill(0f)
        for (i in 0 until dict.nwords()) {
            val word = dict.getWord(i)
            getWordVector(vec, word)
            val norm = vec.norm2()
            if (norm > 0) {
                wordVectors[i] += 1.0f/norm to vec
            }
        }
    }

    /**
     * 把词向量填充到一个Vector对象里面去
     *
     * @param vec
     * @param word
     */
    fun getWordVector(vec: MutableVector, word: String) {
        vec.zero()
        val ngrams = dict.getSubwords(word)
        val buffer = ngrams.buffer
        var i = 0
        val len = ngrams.size()
        while (i < len) {
            addInputVector(vec, buffer[i])
            i++
        }

        if (ngrams.size() > 0) {
            vec *= 1.0f / ngrams.size()
        }
    }

    fun getWordVector(word: String): Vector {
        val vec = MutableByteBufferVector(args.dim)
        getWordVector(vec, word)
        return vec
    }


    /**
     * 计算句子向量
     * @return 句子向量
     */
    fun getSentenceVector(tokens: Iterable<String>): Vector {
        val svec = MutableByteBufferVector(args.dim)
        getSentenceVector(svec, tokens)
        return svec
    }


    /**
     * 句子向量
     *
     * @param svec
     * @param tokens
     */
    private fun getSentenceVector(svec: MutableVector, tokens: Iterable<String>) {
        svec.zero()
        if (args.model == ModelName.sup) {
            val line = IntArrayList()
            val labels = IntArrayList()
            dict.getLine(tokens, line, labels)

            for (i in 0 until line.size()) {
                addInputVector(svec, line.get(i))
            }

            if (!line.isEmpty) {
                svec *= (1.0f / line.size())
            }
        } else {
            val vec = MutableByteBufferVector(args.dim)
            var count = 0
            for (word in tokens) {
                getWordVector(vec, word)
                val norm = vec.norm2()
                if (norm > 0) {
                    vec *= (1.0f / norm)
                    svec += vec
                    count++
                }
            }
            if (count > 0) {
                svec *= (1.0f / count)
            }
        }
    }

    private fun addInputVector(vec: MutableVector, ind: Int) {
        if (quant) {
            model.qinput.addToVector(vec, ind)
        } else {
            vec += input[ind]
        }
    }


    /**
     * 把词向量另存为文本格式
     *
     * @param file
     */
    @Throws(Exception::class)
    fun saveVectors(fileName: String) {
        var fileName = fileName
        if (!fileName.endsWith("vec")) {
            fileName += ".vec"
        }

        val file = File(fileName)
        if (file.exists()) {
            file.delete()
        }
        if (file.parentFile != null) {
            file.parentFile.mkdirs()
        }

        val vec = MutableByteBufferVector(args.dim)
        val df = DecimalFormat("0.#####")

        Files.asByteSink(file).asCharSink(Charsets.UTF_8).openBufferedStream().use { writer ->
            writer.write("${dict.nwords()} ${args.dim}\n")
            for (i in 0 until dict.nwords()) {
                val word = dict.getWord(i)
                getWordVector(vec, word)
                writer.write(word)
                writer.write(" ")
                for (j in 0 until vec.length()) {
                    writer.write(df.format(vec[j].toDouble()))
                    writer.write(" ")
                }
                writer.write("\n")
            }
        }
    }

    /**
     * 保存为自有的文件格式(多文件）
     */
    @Throws(Exception::class)
    fun saveModel(path: String) {
        var path = File(path)
        if (path.exists()) {
            path.deleteRecursively()
        }
        path.mkdirs()

        //dict
        File(path, "dict.bin").outputStream().channel.use {
            dict.save(it)
        }

        //args
        File(path, "args.bin").outputStream().channel.use {
            args.save(it)
        }

        if (!quant) {
            //input float matrix
            File(path, "input.matrix").outputStream().channel.use {
                it.writeInt(model.input.rows())
                it.writeInt(model.input.cols())
                model.input.write(it)
            }
        } else {
            File(path, "qinput.matrix").outputStream().channel.use {
                model.qinput.save(it)
            }
        }

        if (quant && model.quantOut) {
            File(path, "qoutput.matrix").outputStream().channel.use {
                model.qoutput!!.save(it)
            }
        } else {
            File(path, "output.matrix").outputStream().channel.use {
                it.writeInt(model.output.rows())
                it.writeInt(model.output.cols())
                model.output.write(it)
            }
        }
    }


    companion object {

        /**
         * 加载facebook官方C程序保存的文件模型，支持bin和ftz模型
         *
         * @param modelFilePath
         * @throws IOException
         */
        @JvmStatic
        @Throws(Exception::class)
        fun loadFasttextBinModel(modelFilePath: String): FastText {
            return LoadFastTextFromClangModel.loadCModel(modelFilePath)
        }
        /**
         * 加载facebook官方C程序保存的文件模型，支持bin和ftz模型
         *
         * @param modelPath
         * @throws IOException
         */
        @JvmStatic
        @Throws(Exception::class)
        fun loadFasttextBinModel(modelFile: File): FastText {
            return LoadFastTextFromClangModel.loadCModel(modelFile)
        }
        /**
         * 加载facebook官方C程序保存的文件模型，支持bin和ftz模型
         *
         * @param modelPath
         * @throws IOException
         */
        @JvmStatic
        @Throws(Exception::class)
        fun loadFasttextBinModel(modelStream: InputStream): FastText {
            return LoadFastTextFromClangModel.loadCModel(modelStream)
        }

        private fun File.openAutoDataInput() = AutoDataInput.open(this)


        /**
         * 加载java程序保存的文件模型.
         * path应该是一个目录，下面保存各个细节的文件
         */
        @JvmOverloads
        @JvmStatic
        fun loadModel(modelPath: String, mmap: Boolean = true): FastText {
            val dir = File(modelPath)

            if (!dir.exists() || dir.isFile) {
                println("error file $dir")
                exitProcess(0)
            }

            val args = Args().loadClang(File(dir, "args.bin").openAutoDataInput())

            val dictionary = Dictionary(args).load(File(dir, "dict.bin").openAutoDataInput())

            fun loadMatrix(file: File): FloatMatrix {
                return FloatMatrix.loadMatrix(file,mmap)
            }

            val quant = File(dir, "qinput.matrix").exists()

            var input: FloatMatrix = FloatMatrix.floatArrayMatrix(0, 0)
            var qinput: QMatrix? = null

            if (quant) {
                qinput = QMatrix.load(File(dir, "qinput.matrix").openAutoDataInput())
            } else {
                input = loadMatrix(File(dir, "input.matrix"))
            }

            val quantInput = quant
            if (!quantInput && dictionary.isPruned()) {
                throw RuntimeException("Invalid model file.\n"
                        + "Please download the updated model from www.fasttext.cc.\n"
                        + "See issue #332 on Github for more information.\n")
            }

            var output: FloatMatrix = FloatMatrix.floatArrayMatrix(0, 0)
            var qoutput: QMatrix? = null

            val qout = File(dir, "qoutput.matrix").exists()
            if (quant && qout) {
                qoutput = QMatrix.load(File(dir, "qoutput.matrix").openAutoDataInput())
            } else {
                output = loadMatrix(File(dir, "output.matrix"))
            }

            val model = Model(input, output, args, 0)
            if(quantInput){
                model.setQuantizePointer(qinput, qoutput)
            }

            if (args.model == ModelName.sup) {
                model.setTargetCounts(dictionary.getCounts(EntryType.label))
            } else {
                model.setTargetCounts(dictionary.getCounts(EntryType.word))
            }


            return FastText(args, dictionary,  model)
        }



        @JvmOverloads
        @Throws(Exception::class)
        @JvmStatic
        fun train(trainFile: File, model_name: ModelName = ModelName.sup, args: TrainArgs = TrainArgs()): FastText {
            return FastTextTrain().train(trainFile, model_name, args)
        }



        /**
         * 分类模型量化
         *
         * @param out
         */
        fun quantize(fastText: FastText,
                     dsub:Int=2,
                     qnorm:Boolean=false):FastText {

            if (fastText.quant) {
                println("该模型已经被量化过")
                return fastText
            }

            if(fastText.args.model != ModelName.sup){
                throw RuntimeException("Only for sup model")
            }

            val qMatrix = QMatrix(fastText.input.rows(),fastText.input.cols(), dsub, qnorm)
            val inputMatrix = fastText.input.toMutableFloatMatrix()
            qMatrix.quantize(inputMatrix)


            val qModel = Model(FloatMatrix.floatArrayMatrix(0, 0),fastText.output,fastText.args,0)

            qModel.setQuantizePointer(qMatrix,null)


            val QFastText = FastText(fastText.args,fastText.dict,qModel)

            return QFastText
        }
    }
}

class Model(val input: FloatMatrix
            , val output: FloatMatrix,
            args_: Args,
            seed: Int) : BaseModel(args_, seed, output.rows()) {

    /**
     * 是否乘积量化模型(input)
     */
    var quant: Boolean = false

    /**
     * Right 是否量化
     */
    var quantOut = false

    var qinput = QMatrix()
    var qoutput = QMatrix()

    /**
     * hidden size 也就是向量的维度
     */
    private val hsz: Int = args_.dim // dim

    private val comparePairs = { o1: FloatIntPair, o2: FloatIntPair -> Floats.compare(o2.first, o1.first) }

    fun std_log(d: Float)=Math.log(d+1e-5)


    fun setQuantizePointer(qinput: QMatrix?, qoutput: QMatrix?) {

        qinput?.let {
            quant = true
            this.qinput = qinput
        }
        // qoutput 不为null就是out向量化
        qoutput?.let {
            quantOut = true
            this.qoutput = it
            this.outputMatrixSize = qoutput.m
        }
    }

    fun predict(input: IntArrayList, k: Int,
                heap: MutableList<FloatIntPair>,
                hidden: MutableVector,
                output: MutableVector) {
        checkArgument(k > 0)

        computeHidden(input, hidden)
        if (args_.loss == LossName.hs) {
            dfs(k, 2 * outputMatrixSize - 2, 0.0f, heap, hidden)
        } else {
            findKBest(k, heap, hidden, output)
        }
        Collections.sort(heap, comparePairs)
    }

    fun findKBest(k: Int, heap: MutableList<FloatIntPair>, hidden: Vector, output: MutableVector) {
        computeOutputSoftmax(hidden, output)
        for (i in 0 until outputMatrixSize) {
            val logoutputi = std_log(output[i]).toFloat()
            if (heap.size == k && logoutputi < heap[heap.size - 1].first) {
                continue
            }
            heap.add(FloatIntPair(logoutputi, i))
            Collections.sort(heap, comparePairs)
            if (heap.size > k) {
                Collections.sort(heap, comparePairs)
                heap.removeAt(heap.size - 1) // pop last
            }
        }
    }

    fun dfs(k: Int, node: Int, score: Float, heap: MutableList<FloatIntPair>, hidden: Vector) {
        if (heap.size == k && score < heap[heap.size - 1].first) {
            return
        }

        if (tree[node].left == -1 && tree[node].right == -1) {
            heap.add(FloatIntPair(score, node))
            Collections.sort(heap, comparePairs)
            if (heap.size > k) {
                Collections.sort(heap, comparePairs)
                heap.removeAt(heap.size - 1) // pop last
            }
            return
        }

//        val f = sigmoid(output.dotRow(hidden, node - outputMatrixSize))
        var f = if (quant && quantOut) {
            qoutput.dotRow(hidden, node - outputMatrixSize)
        } else {
            output[node - outputMatrixSize] * hidden
        }
        f = 1.0f / (1 + exp(-f))


        dfs(k, tree[node].left, score + std_log(1.0f - f).toFloat(), heap, hidden)
        dfs(k, tree[node].right, score + std_log(f).toFloat(), heap, hidden)
    }


    private fun computeHidden(input: IntArrayList, hidden: MutableVector) {
        checkArgument(hidden.length() == hsz)
        hidden.zero()

        val buffer = input.buffer
        var i = 0
        val size = input.size()
        while (i < size) {
            val it = buffer[i]
            if (quant) {
                qinput.addToVector(hidden, it)
            } else {
                hidden += this.input[it]
            }
            i++
        }
        hidden *= (1.0f / input.size())
    }

    private fun computeOutputSoftmax(hidden: Vector, output: MutableVector) {
        if (quant && quantOut) {
            matrixMulVector(qoutput, hidden, output)
        } else {
            matrixMulVector(this.output, hidden, output)
        }

        var max = output[0]
        var z = 0.0f
        for (i in 1 until outputMatrixSize) {
            max = Math.max(output.get(i), max)
        }
        for (i in 0 until outputMatrixSize) {
            output[i] = Math.exp((output[i] - max).toDouble()).toFloat()
            z += output[i]
        }
        for (i in 0 until outputMatrixSize) {
            output[i] = output[i] / z
        }
    }

    private fun matrixMulVector(matrix: QMatrix, v: Vector, target: MutableVector) {
        checkArgument(matrix.m == target.length())
        checkArgument(matrix.n == v.length())

        val m_ = matrix.m
        for (i in 0 until m_) {
            target[i] = matrix.dotRow(v,i)
        }
    }

}

