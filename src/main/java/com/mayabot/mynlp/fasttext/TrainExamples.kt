package com.mayabot.mynlp.fasttext

import com.google.common.base.CharMatcher
import com.google.common.base.Splitter
import com.google.common.collect.Lists
import java.io.BufferedReader
import java.io.File


/**
 * 基于内存的实现。提供add的方法，全部在内存里面。
 */
class MemTrainExampleSource(val splitter: WordSplitter) : TrainExampleSource{

    var list:MutableList<List<String>> = Lists.newArrayList<List<String>>()

    fun addExample(text: String) {
        list.add(splitter.split(text))
    }

    fun addExample(text: String,label:String) {
        var split = ArrayList<String>().apply {
            addAll(splitter.split(text))
            add(label)
        }
        list.add(split)
    }

    override fun iteratorAll(): ExampleIterator {
        var iterator = list.iterator()
        return object : ExampleIterator {
            override fun close() {
            }
            override fun hasNext(): Boolean {
                return iterator.hasNext()
            }
            override fun next(): List<String> {
                return iterator.next()
            }

            override fun remove() {
            }
        }
    }

    override fun split(num: Int): List<TrainExampleSource> {
        return Lists.partition(list,num.toInt()).map {
            val x = MemTrainExampleSource(splitter)
            x.list = it
            x
        }.toList()

    }

    override fun close() {

    }
}


/**
 * 基于文件，一行一个Example
 */
class FileTrainExampleSource(val splitter: WordSplitter,
                          val file:File): TrainExampleSource{

    var subFiles: List<File>? = null

    override fun iteratorAll(): ExampleIterator {

        return object : AbstractIterator<List<String>>(),ExampleIterator{

            val bufferedReader : BufferedReader = file.bufferedReader(Charsets.UTF_8)

            override fun computeNext() {
                val line = bufferedReader.readLine()
                if (line == null) {
                    done()
                }else{
                    setNext(splitter.split(line))
                }
            }

            override fun close() {
                bufferedReader.close()
            }

            override fun remove() {
            }
        }
    }

    override fun split(num: Int): List<TrainExampleSource> {
        val dir = file.parentFile
        val fileName = file.name

        val subFiles = (1 .. num).map { File(dir,fileName+"_"+it) }
        this.subFiles = subFiles
        val subFileWriter = subFiles.map { it.bufferedWriter(Charsets.UTF_8) }

        var count = 0
        iteratorAll().forEach {
            subFileWriter[count%num].append(
                    it.joinToString(separator = " ",postfix = "\n")
            )
            count++
        }

        subFileWriter.forEach {
            it.flush()
            it.close()
        }

        return subFiles.map { FileTrainExampleSource(whitespaceSplitter,it) }.toList()
    }

    override fun close() {
        subFiles?.forEach {
            it.delete()
        }
    }

}




/**
 * 默认就是这么实现的
 */
object whitespaceSplitter : WordSplitter{

    val whitespace = Splitter.on(CharMatcher.whitespace())


    override fun split(text: String): List<String> {
        return whitespace.splitToList(text)
    }
}
