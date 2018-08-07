package com.mayabot.mynlp.fasttext

fun main(args: Array<String>) {
    val train = FastText.loadModel("/Users/jimichan/Downloads/model0726.dir")

    val text = listOf(
    "^ 霸王腰花 是 什么 $",
    "^ 洗碗机 开 门 $",
    "^ 你好 吗 $",
    "^ 搞 个 三 菜 一 汤 $",
    "^ 还 要 多 久 $",
    "^ 播放 一首 刘德华 的 歌 $",
    "^ 今天 北京 天气 怎么样 $",
    "^ 启动 汽车 引擎 $",
    "^ 找 一家 中国餐厅 $",
    "^ 鱼翅 怎么做 $",
    "^ 有什么 清热解暑 的 菜 吗 $")
    //train.saveModel("/Users/jimichan/Downloads/model0726.dir")
    //println(train.predict("^ 霸王腰花 是 什么 $".split(" "),10))
    //println(train.predict("^ 洗碗机 开 门 $".split(" "),10))
    text.forEach { line->
        println("$line \t\t"+train.predict(line.split(" "),10))
    }

}
