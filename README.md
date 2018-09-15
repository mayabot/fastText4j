## FastText4j

   Implementing Facebook's FastText with java. [Fasttext](https://github.com/facebookresearch/fastText/) is a library for text representation and classification by facebookresearch. It implements text classification and word embedding learning.
   
Features:

 * 100% in java
 * Compatible with original C++ model file (include quantizer compression model)
 * Provides training api (almost the same performance)
 * Support for java file formats( can read file use mmap),read big model file with less memory
 * Well-designed API
 
## Installing

### Gradle
```
compile 'com.mayabot:fastText4j:1.2.2'
```

### Maven
```xml
<dependency>
  <groupId>com.mayabot</groupId>
  <artifactId>fastText4j</artifactId>
  <version>1.2.2</version>
</dependency>
```

## Tutorial

### 1. Train model
![GitHub](https://cdn.mayabot.com/nlp/wiki-images/fast_train.png "GitHub,Social Coding")

- ModelName.sup supervised
- ModelName.sg   skipgram
- ModelName.cow cbow

```java
//Word representation learning
FastText fastText = FastText.train(new File("train.data"), ModelName.sg);

// Text classification

FastText fastText = FastText.train(new File("train.data"), ModelName.sup);

```

data.txt is also encoded in utf-8 with one sample each line. And it needs to do word spliting beforehand as well. There is a string starting with ```__label__``` in each line，representing the classifying target, such as ```__label__正面```. Each sample could have  multiple label. Through the attribute 'label' in TrainArgs, you can customise the head.

### 2. save model

save model to java format
```java
fastText.saveModel("path/data.model");
```

### 3. load model

public Fasttext loadModel(String modelPath, boolean mmap)

```java
//load from java format 
FastText fastText = FastText.loadModel("path/data.model",true);

//load from c++ format
FastText fastText = FastText.loadFasttextBinModel("path/wiki.bin") 

```

### 4. quantizer compression
 FastText quantize(FastText fastText , int dsub=2, boolean qnorm=false)
```java
//load from java format 
FastText quantizerFastText = FastText.quantize(fastText,2,false);
```


### 5.Predict
```java
//predict the result of a word
List<FloatStringPair> predict = fastText.predict(Arrays.asList("fastText在预测标签时使用了非线性激活函数".split(" ")), 5);
```

### 6.Nearest Neighbor Search
```java
List<FloatStringPair> predict = fastText.nearestNeighbor("中国",5);
```

### 7.Analogies
By giving three words A, B and C, return the nearest words in terms of semantic distance and their similarity list, under the condition of (A - B + C).
```java
List<FloatStringPair> predict = fastText.analogies("国王","皇后","男",5);
```


### Parameters of TrainArgs

The parameters is consistant with the C++ version :
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

## Resource
### Official pre-trained model
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

fastText is BSD-licensed. [facebook provide an additional patent grant.](https://github.com/facebookresearch/fastText/blob/master/PATENTS)
