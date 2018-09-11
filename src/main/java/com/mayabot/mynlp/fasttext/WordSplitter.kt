package com.mayabot.mynlp.fasttext

/**
 *  如果分词器不能把 __lable__xxxx 分为一个词，那么要特殊处理一下
 */
interface WordSplitter {
    fun split(text:String) : List<String>
}