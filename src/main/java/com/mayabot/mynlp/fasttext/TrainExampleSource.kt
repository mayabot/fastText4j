package com.mayabot.mynlp.fasttext

interface TrainExampleSource {

    fun iteratorAll() : ExampleIterator

    fun split(num:Int) : List<TrainExampleSource>

    fun close()
}