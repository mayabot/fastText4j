## 介绍

 [Fasttext](https://github.com/facebookresearch/fastText/)是facebookresearch的一个text representation and classification C++程序库。这个库提供两个功能，文本分类和词嵌入学习。

## mynlp-fasttext

 在mynlp中提供一个fasttext java版本的实现，使用kotlin编写。有如下特征:

 * 100% 纯java实现
 * 兼容原版模型文件

   fasttext官方提供各种预先训练的模型，可以直接读取
 * 兼容原版乘积量化压缩模型
 * java版本也提供训练API（性能与原版相当）
 * 支持私有的存储格式
 * 在私有存储格式里，支持mmap读取模型文件

   官方提供的中文wiki模型大小为2.8G，需要jvm至少4G才能运行，需要加载时间也很长。通过mmap方式，只需少量内存，在3秒左右即可加载完毕模型文件。

## Intalling

目前还没有发布到maven中央仓库，在mayabot的公开仓库中

在Gradle增加一个maven仓库地址 https://nexus.mayabot.com/content/groups/public/

### Gradle
```
repositories {
        maven {
            url = "https://nexus.mayabot.com/content/groups/public/"
        }
}

dependencies {
    compile 'com.mayabot:fastText4j:1.1.0'
}

```

### Maven
```xml

<repositories>
    <repository>
        <id>mayabot-public</id>
        <url>http://nexus.mayabot.com/content/groups/public/</url>
    </repository>
</repositories>
	
	
<dependency>
  <!-- mynlp-fasttest @ https://mynlp.info/ -->
  <groupId>com.mayabot</groupId>
  <artifactId>fastText4j</artifactId>
  <version>1.1.0</version>
</dependency>
```

## Example use cases

### 1.词向量表示学习
```java
File file = new File("data/fasttext/data.text");

FastText fastText = FastText.train(file, ModelName.sg);

fastText.saveModel("data/fasttext/model.bin");
```
data.txt是训练文件，采用utf-8编码存储。训练文本中词需要预先分词，采用空格分割。默认设置下，采用3-6的char ngram。
除了sg算法，你还可以采用cow算法。如果需要更多的参数设置，请提供TrainArgs对象进行设置。

### 2.分类模型训练
```java
File file = new File("data/fasttext/data.txt");

FastText fastText = FastText.train(file, ModelName.sup);

fastText.saveModel("data/fasttext/model.bin");
```
data.txt同样也是utf-8编码的文件，每一行一个example，同样需要预先分词。每一行中存在一个```__label__```为前缀的字符串，表示该example的分类目标，比如```__label__正面```，每个example可以存在多个label。你可以设置TrainArgs中label属性，指定自定义的前缀。
获得模型后，可以通过predict方法进行分类结果预测。



### 3.加载官方模型文件，另存为java模型格式
```java
FastText fastText = FastText.loadFasttextBinModel("data/fasttext/wiki.zh.bin");
fastText.saveModel("data/fasttext/wiki.model");
```

### 4.分类预测
```java
//predict传入一个分词后的结果
FastText fastText = FastText.loadCModel("data/fasttext/wiki.zh.bin");
List<FloatStringPair> predict = fastText.predict(Arrays.asList("fastText在预测标签时使用了非线性激活函数".split(" ")), 5);
```

### 5.Nearest Neighbor 近邻查询
```java
FastText fastText = FastText.loadCModel("data/fasttext/wiki.zh.bin");

List<FloatStringPair> predict = fastText.nearestNeighbor("中国",5);
```

### 6.Analogies 类比
给定三个词语A、B、C，返回与(A - B + C)语义距离最近的词语及其相似度列表。
```java
FastText fastText = FastText.loadCModel("data/fasttext/wiki.zh.bin");

List<FloatStringPair> predict = fastText.analogies("国王","皇后","男",5);
```

## Api
```java

 /**
 * 在sup模型上，预测分类label
 */
 List<FloatStringPair> predict(Iterable<String> tokens, k: Int)

 /**
 * 近邻搜索(相似词搜索)
 * @param word 
 * @param k k个最相似的词
 */
 List<FloatStringPair> nearestNeighbor(String word, k: Int)

/**
 * 类比搜索
 * Query triplet (A - B + C)?
 */
 List<FloatStringPair> analogies(String A,String B,String C, k: Int)

 /**
 * 查询指定词的向量
 */
 Vector getWordVector(String word)

 /**
 * 获得短语的向量表示
 */
 Vector getSentenceVector(Iterable<String> tokens)

 /**
 * 保存词向量为文本格式
 */
 saveVectors(String fileName)

 /**
 * 保存模型为二进制格式
 */
 saveModel(String file)

 /**
 * 训练一个模型
 * @param File trainFile
 * @param model_name
 *  sg skipgram 词向量之skipgram算法
 *  cow cbow 词向量之cbow算法
 *  sup supervised 文本分类
 * @param args 训练参数 
 **/
 FastText FastText.train(File trainFile, ModelName model_name, TrainArgs args)

 /**
 * 加载有saveModel方法保存的模型
 * @param file 
 * @param mmap 是否采用mmap加载模型文件，可以在有限内存下，快速加载大模型文件
 */
 Fasttext.loadModel(String file,boolean mmap)


 /**
 * 加载facebook官方C程序保存的文件模型，支持bin和ftz模型
 */
 Fasttext.loadFasttextBinModel(String binFile)
```


## TrainArgs和相关参数
java版本的参数和C++版本的保持一致，参考如下：
```
The following arguments for the dictionary are optional:
  -minCount           minimal number of word occurences [1]
  -minCountLabel      minimal number of label occurences [0]
  -wordNgrams         max length of word ngram [1]
  -bucket             number of buckets [2000000]
  -minn               min length of char ngram [0]
  -maxn               max length of char ngram [0]
  -t                  sampling threshold [0.0001]
  -label              labels prefix [__label__]

The following arguments for training are optional:
  -lr                 learning rate [0.1]
  -lrUpdateRate       change the rate of updates for the learning rate [100]
  -dim                size of word vectors [100]
  -ws                 size of the context window [5]
  -epoch              number of epochs [5]
  -neg                number of negatives sampled [5]
  -loss               loss function {ns, hs, softmax} [softmax]
  -thread             number of threads [12]
  -pretrainedVectors  pretrained word vectors for supervised learning []
  -saveOutput         whether output params should be saved [0]

The following arguments for quantization are optional:
  -cutoff             number of words and ngrams to retain [0]
  -retrain            finetune embeddings if a cutoff is applied [0]
  -qnorm              quantizing the norm separately [0]
  -qout               quantizing the classifier [0]
  -dsub               size of each sub-vector [2]
```

## 资源
### 官方预训练模型
Recent state-of-the-art [English word vectors](https://fasttext.cc/docs/en/english-vectors.html).<br/>
Word vectors for [157 languages trained on Wikipedia and Crawl](https://github.com/facebookresearch/fastText/blob/master/docs/crawl-vectors.md).<br/>
Models for [language identification](https://fasttext.cc/docs/en/language-identification.html#content) and [various supervised tasks](https://fasttext.cc/docs/en/supervised-models.html#content).

## References

Please cite [1](#enriching-word-vectors-with-subword-information) if using this code for learning word representations or [2](#bag-of-tricks-for-efficient-text-classification) if using for text classification.

### Enriching Word Vectors with Subword Information

[1] P. Bojanowski\*, E. Grave\*, A. Joulin, T. Mikolov, [*Enriching Word Vectors with Subword Information*](https://arxiv.org/abs/1607.04606)

```
@article{bojanowski2017enriching,
  title={Enriching Word Vectors with Subword Information},
  author={Bojanowski, Piotr and Grave, Edouard and Joulin, Armand and Mikolov, Tomas},
  journal={Transactions of the Association for Computational Linguistics},
  volume={5},
  year={2017},
  issn={2307-387X},
  pages={135--146}
}
```

### Bag of Tricks for Efficient Text Classification

[2] A. Joulin, E. Grave, P. Bojanowski, T. Mikolov, [*Bag of Tricks for Efficient Text Classification*](https://arxiv.org/abs/1607.01759)

```
@InProceedings{joulin2017bag,
  title={Bag of Tricks for Efficient Text Classification},
  author={Joulin, Armand and Grave, Edouard and Bojanowski, Piotr and Mikolov, Tomas},
  booktitle={Proceedings of the 15th Conference of the European Chapter of the Association for Computational Linguistics: Volume 2, Short Papers},
  month={April},
  year={2017},
  publisher={Association for Computational Linguistics},
  pages={427--431},
}
```

### FastText.zip: Compressing text classification models

[3] A. Joulin, E. Grave, P. Bojanowski, M. Douze, H. Jégou, T. Mikolov, [*FastText.zip: Compressing text classification models*](https://arxiv.org/abs/1612.03651)

```
@article{joulin2016fasttext,
  title={FastText.zip: Compressing text classification models},
  author={Joulin, Armand and Grave, Edouard and Bojanowski, Piotr and Douze, Matthijs and J{\'e}gou, H{\'e}rve and Mikolov, Tomas},
  journal={arXiv preprint arXiv:1612.03651},
  year={2016}
}
```

(\* These authors contributed equally.)

## License

fastText is BSD-licensed. [Facebook持有专利](https://github.com/facebookresearch/fastText/blob/master/PATENTS)
