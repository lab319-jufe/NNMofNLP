
<!-- TOC -->

- [1. 编码分类特征](#1-编码分类特征)
    - [1.1. 独热编码](#11-独热编码)
        - [1.1.1. 分类变量的的独热编码类似于虚拟变量/哑变量，例：](#111-分类变量的的独热编码类似于虚拟变量哑变量例)
        - [1.1.2. 文本分析中的独热编码](#112-文本分析中的独热编码)
    - [1.2. 稠密编码（特征嵌入）](#12-稠密编码特征嵌入)
- [2. 组合稠密向量](#2-组合稠密向量)
- [3. 独热和稠密向量间的关系](#3-独热和稠密向量间的关系)
- [4. 杂项](#4-杂项)
- [5. 例子：词性标注](#5-例子词性标注)
- [6. 例子：弧分解分析](#6-例子弧分解分析)

<!-- /TOC -->

# 1. 编码分类特征

## 1.1. 独热编码

### 1.1.1. 分类变量的的独热编码类似于虚拟变量/哑变量，例：

性别：男、女
宠物：猫、狗、其他
区域：都会、城镇、乡村、自然保护区

|性别|宠物|区域|
|-|-|-|
|0|2|3|
|1|1|0|
|0|2|1|
|1|0|2|

第一个人经过独热编码：

|性别0|性别1|宠物0|宠物1|宠物2|区域0|区域1|区域2|区域3|
|-|-|-|-|-|-|-|-|-|
|1|0|0|0|1|0|0|0|1|

```python{cmd=True}
from sklearn import preprocessing
enc = preprocessing.OneHotEncoder()
enc.fit([[0,2,3],[1,1,0],[0,2,1],[1,0,2]])

array = enc.transform([[0,2,3]]).toarray()

print(array)
```

```output
[[1. 0. 0. 0. 1. 0. 0. 0. 1.]]
```

### 1.1.2. 文本分析中的独热编码

```python{cmd=True}
import jieba
#定义停用词、标点符号
punctuation = ["，","。", "：", "；", "？"]
#定义语料
content = ["机器学习带动人工智能飞速的发展。",
            "深度学习带动人工智能飞速的发展。",
            "机器学习和深度学习带动人工智能飞速的发展。"
            ]
#分词
segs_1 = [jieba.lcut(con) for con in content]
print(segs_1)
```

```output
[['机器', '学习', '带动', '人工智能', '飞速', '的', '发展', '。'], ['深度', '学习', '带动', '人工智能', '飞速', '的', '发展', '。'], ['机
器', '学习', '和', '深度', '学习', '带动', '人工智能', '飞速', '的', '发展', '。']]
```

下面操作就是把所有的分词结果放到一个袋子（List）里面，也就是取并集，再去重，获取对应的特征词。

```python {cmd=True}
#求并集
bag_of_words = [ x for item in segs_1 for x in item if x not in punctuation]
#去重
bag_of_words = list(set(bag_of_words))
print(bag_of_words)
```

```output
['的', '发展', '人工智能', '深度', '带动', '机器', '飞速', '学习', '和']
```

我们以上面特征词的顺序，完成词袋化，最后得到词袋向量：

``` python{cmd=True}
bag_of_word2vec = []
for sentence in tokenized:
    tokens = [1 if token in sentence else 0 for token in bag_of_words ]
    bag_of_word2vec.append(tokens)
print(bag_of_word2vec)
```

```output
[[1, 1, 1, 0, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 0, 1, 1, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1]]
```

## 1.2. 稠密编码（特征嵌入）

- 自然语言处理中，独热编码的结果会变得非常稀疏，拥有2000个词汇的词表会出现2000个维度

- 基于前馈神经网络的 NLP 一般的编码步骤：
  1. 抽取一组和预测输出类别相关的核心语言学特征$f_{1}, \cdots, f_{k}$。
  2. 对于每一个感兴趣的$$f_{i}$，检索 出相应的向量$v\left(f_{i}\right)$ 。
  3. 将特征向量组合成（拼接、相加或者两者组合）输入向量 x 。
  4. 将 x 输入到非线性分类器中（前馈神经网络） 。
  
- 参考
  - 使用稠密编码的神经网络：[Chen 和 Manning(2014)](https://cs.stanford.edu/~danqi/papers/emnlp2014.pdf)、 [Collobert 和 Weston(2008)](http://www.thespermwhale.com/jaseweston/papers/unified_nlp.pdf))、 [Collobert 等人(2011)](http://www.jmlr.org/papers/volume12/collobert11a/collobert11a.pdf)

  - 使用稀疏编码的神经网络：[Johnson Zhang(2015)](https://www.aclweb.org/anthology/N15-1011)

# 2. 组合稠密向量

# 3. 独热和稠密向量间的关系

# 4. 杂项

# 5. 例子：词性标注

# 6. 例子：弧分解分析
