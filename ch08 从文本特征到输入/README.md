<h1>
第八章 从文本特征到输入
</h1>

<!-- TOC -->

- [1. 编码分类特征](#1-编码分类特征)
    - [1.1. 独热编码](#11-独热编码)
        - [1.1.1. 分类变量的的独热编码类似于虚拟变量/哑变量，例：](#111-分类变量的的独热编码类似于虚拟变量哑变量例)
        - [1.1.2. 文本分析中的独热编码](#112-文本分析中的独热编码)
    - [1.2. 稠密编码（特征嵌入）](#12-稠密编码特征嵌入)
- [2. 组合稠密向量](#2-组合稠密向量)
    - [2.1. 基于窗口的特征](#21-基于窗口的特征)
    - [2.2. 可变特征数目：连续词袋](#22-可变特征数目连续词袋)
- [3. 独热和稠密向量间的关系](#3-独热和稠密向量间的关系)
- [4. 杂项](#4-杂项)
    - [4.1. 距离与位置特征](#41-距离与位置特征)
    - [4.2. 补齐 、 未登录词和词丢弃](#42-补齐--未登录词和词丢弃)
    - [4.3. 特征组合](#43-特征组合)
    - [4.4. 向量共享](#44-向量共享)
    - [4.5. 维度](#45-维度)
    - [4.6. 嵌入的词表](#46-嵌入的词表)
    - [4.7. 网络的输出](#47-网络的输出)
- [5. 例子：词性标注](#5-例子词性标注)
- [6. 例子：弧分解分析(句法分析)](#6-例子弧分解分析句法分析)

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

```python{cmd=True hide=True id='pre'}
import io
import sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')
```

```python{cmd=True id='onehot' continue='pre'}
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

```python {cmd=True continue='onehot' id='onehot2'}
#求并集
bag_of_words = [ x for item in segs_1 for x in item if x not in punctuation]
#去重
bag_of_words = list(set(bag_of_words))
bag_of_words.sort()
print(bag_of_words)
```

```output
['人工智能', '发展', '和', '学习', '带动', '机器', '深度', '的', '飞速']
```

我们以上面特征词的顺序，完成词袋化，最后得到词袋向量：

``` python{cmd=True continue='onehot2' id='onehot3'}
bag_of_word2vec = []
for sentence in segs_1 :
    tokens = [1 if token in sentence else 0 for token in bag_of_words ]
    bag_of_word2vec.append(tokens)
print(bag_of_word2vec)
```

```output
[[1, 1, 0, 1, 1, 1, 0, 1, 1], [1, 1, 0, 1, 1, 0, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1]]
```

## 1.2. 稠密编码（特征嵌入）

- 自然语言处理中，独热编码的结果会变得非常稀疏，例如拥有2000个词汇的词表会出现2000个维度

- 基于前馈神经网络的 NLP 一般的编码步骤：
  1. 抽取一组和预测输出类别相关的核心语言学特征$f_{1}, \cdots, f_{k}$。
  2. 对于每一个感兴趣的$f_{i}$，检索 出相应的向量$v\left(f_{i}\right)$ 。
  3. 将特征向量组合成（拼接、相加或者两者组合）输入向量 x 。
  4. 将 x 输入到非线性分类器中（前馈神经网络） 。
  
- 参考
  - 使用稠密编码的神经网络：[Chen 和 Manning(2014)](https://cs.stanford.edu/~danqi/papers/emnlp2014.pdf)、 [Collobert 和 Weston(2008)](http://www.thespermwhale.com/jaseweston/papers/unified_nlp.pdf))、 [Collobert 等人(2011)](http://www.jmlr.org/papers/volume12/collobert11a/collobert11a.pdf)

  - 使用稀疏编码的神经网络：[Johnson Zhang(2015)](https://www.aclweb.org/anthology/N15-1011)

# 2. 组合稠密向量

## 2.1. 基于窗口的特征

- 窗口大小$k=2$，得到词 a,b,c,d，对应词向量 $a,b,c,d$ 。
  - 若关心相对位置，则使用拼接
  $$\left[\begin{array}{llll}{a} ; & {b} ;& {c} ;& {d}\end{array}\right]$$
  - 若不关心相对位置，只关心与中心词的距离，则使用加权求和
  $$\frac{1}{2} a+b+c+\frac{1}{2} d$$
  - 两者结合，关心在中心词前还是后，但是不关心词距
  $$[(a+b) ;(c+d)]$$

## 2.2. 可变特征数目：连续词袋

目的：使用固定大小的向量表示任意数量的特征

- **连续词袋（CBOW）**([Mikolov，2013b](https://arxiv.org/pdf/1309.4168.pdf))，类似于加权求和的传统词袋
  
$$\mathrm{CBOW}\left(f_{1}, \cdots, f_{k}\right)=\frac{1}{k} \sum_{i=1}^{k} v\left(f_{i}\right)$$

$$
\mathrm{WCBOW}\left(f_{1}, \cdots, f_{k}\right)=\frac{1}{\sum_{i=1}^{k} a_{i}} \sum_{i=1}^{k} a_{i} v\left(f_{i}\right)
$$

$a_{i}$表示特征$f_{i}$的权重，如果$f_{i}$是一个词，则$a_{i}$可以是tf-idf值

# 3. 独热和稠密向量间的关系

d维稠密向量 $v(f_{i})$ 可以看作是由特征值 $f_{i}$ 经过**嵌入层或查找层**映射得到的，嵌入矩阵$\boldsymbol{E}$维度为$|V| \times d$
$$
v\left(f_{i}\right)=f_{i} \boldsymbol{E}
$$
$$
\mathrm{CBOW}\left(f_{1}, \cdots, f_{k}\right)=\sum_{i=1}^{k}\left(f_{i} \boldsymbol{E}\right)=\left(\sum_{i=1}^{k} f_{i}\right) \boldsymbol{E}
$$

类似的，神经网络的第一层可以表示为

$$
\begin{array}{l}{x W+b=\left(\sum_{i=1}^{k} f_{i}\right) W+b} \\ {W \in \mathbb{R}^{|V| \times d}, \quad b \in \mathbb{R}^{d}}\end{array}
$$

其中$W$即为嵌入矩阵

# 4. 杂项

## 4.1. 距离与位置特征

- **含义**：两个词之间隔了多少个其他的词
  - 事件抽取任务
  预测元素词是否为该触发词所代表的事件的一个元素，如：
    - {**触发词**：元素词}
    - {**恐怖袭击**：爆炸，汽车炸弹，枪击，骚乱}
    - {**购物**：商场，服饰，零食}
  - 共指消解任务
  判断代词 he 或者 she 指代的具体内容
  
- **处理方法**：将距离作为特征，分组（如$1,2,3,4,5-10,10+$），one-hot编码，之后的步骤与其他特征一致

## 4.2. 补齐 、 未登录词和词丢弃

>**补齐**
**未登录词** 未登录词即没有被收录在分词词表中但必须切分出来的词，包括各类专有名词（人名、地名、企业名等）、缩写词、新增词汇等等
**词签名** 处理未登录词的另 一种技术是将词的形式回退到词签名。比如用*-ing* 符号代替以 ing 结尾的未登录词
**词丢弃** 在训练集中抽取特征时，用未登录符号随机替换单词
**使用词丢弃进行正则化**

## 4.3. 特征组合

- 线性方法
- 核方法

## 4.4. 向量共享

- 不同位置的同一词汇是否应该拥有相同的词向量，目前仍没有定论

## 4.5. 维度

## 4.6. 嵌入的词表

- 训练使用的词表只基于训练集，需要为不在词表中的所有词赋予特殊的向量

## 4.7. 网络的输出

- 在多分类任务中，神经网络输出的结果是 k 个分类的得分
- 在此之前得到的是 d 维稠密向量到 k 维输出层的 $d\times k$维矩阵
- 该矩阵向量表示之间的相关性指示了模型输出类别之间的相关性

# 5. 例子：词性标注

编码的工程量有点大，只以 **jieba**为例，介绍如何调库，对中文文本标注词性

```python {cmd=True continue='pre'}
import jieba.posseg as pseg
string = '''随着农村脱贫战略的愈演愈烈，城市贫困问题也逐步进入公众的眼界。
相对于农村贫困的多年深度研究，我国城镇贫困的测度还处在较为青涩的阶段，
各方面仍存在许多不足，也正由于农村脱贫的日益完善和城镇贫困的不完善使得
城市贫困测度亟待解决。因此，本文选择使用来自北京大学“985”项目资助、北京
大学中国社会科学调查中心执行的中国家庭追踪调查（CFPS）的数据对城市贫困进行研究。'''
seg = pseg.cut(string)
for word, flag in seg:
    print('%s %s' % (word, flag))

```

```output
Building prefix dict from the default dictionary ...
Loading model from cache C:\Users\zhuqian\AppData\Local\Temp\jieba.cache
Loading model cost 2.225 seconds.
Prefix dict has been built succesfully.
随着 p
农村 n
脱贫 v
战略 n
的 uj
愈演愈烈 l
， x
城市 ns
贫困 a
问题 n
也 d
逐步 d
进入 v
公众 n
的 uj
眼界 n
。 x
相对 d
于 p
农村 n
贫困 a
的 uj
多年 m
深度 ns
研究 vn
， x
```

由于汉语独有的特点，jieba的词性标注和分词同时进行

![jieba流程](https://images2015.cnblogs.com/blog/668850/201612/668850-20161212140153839-1228771950.png)

- **源码**(jieba/posseg)：
  - \_\_init__.py实现了词性标注的大部分**函数**；
  - char_state_tab.py存储了**离线统计**的字及其对应的状态；
  - prob_emit.py存储了状态到字的**发射概率**的对数值；
  - prob_start.py存储了**初始状态**的概率的对数值；
  - prob_trans.py存储了前一时刻的状态到当前时刻的状态的**转移概率**的对数值；
  - viterbi.py实现了**Viterbi算法**；

# 6. 例子：弧分解分析(句法分析)

![句子上面画弧](https://pic3.zhimg.com/80/ebf2cc1f85b1b191d810434ef61578fd_hd.jpg)

@import "StanfordCoreNlp.py"

```output
(ROOT
  (IP
    (NP
      (CP
        (IP
          (NP (NN 项目组))
          (VP (VV 结合)
            (NP
              (NP (NR 北京))
              (NP (NN 教科) (NN 院)))))
        (DEG 的))
      (QP (CD 一些))
      (NP (NN 要求)))
    (PU ，)
    (VP
      (VP (VV 讨论) (AS 了)
        (NP
          (QP (CD 一)
            (CLP (M 个)))
          (DNP
            (ADJP (JJ 初步))
            (DEG 的))
          (NP (NN 框架))))
      (PU ，)
      (IP
        (IP
          (PP (P 随着)
            (NP
              (NP
                (DNP
                  (NP (NN 数据))
                  (DEG 的))
                (NP (NN 更新)))
              (CC 以及)
              (NP
                (CP
                  (IP
                    (VP
                      (ADVP (AD 刚))
                      (VP (VV 开始)
                        (NP
                          (QP (CD 几)
                            (CLP (M 份)))
                          (NP (NN 报告))))))
                  (DEG 的))
                (NP (NN 总结)))))
          (PU ，)
          (VP (VV 分析)
            (IP
              (NP (NN 框架))
              (VP
                (ADVP (AD 逐渐))
                (VP (VV 得到) (AS 了)
                  (NP (NN 完善)))))))
        (PU ，)
        (IP
          (ADVP (AD 最终))
          (VP
            (VRD (VV 演变) (VV 成))
            (NP
              (DNP
                (NP (NN 下图))
                (DEC 的))
              (NP (NN 分析) (NN 框架)))))))
    (PU 。)))
[('ROOT', 0, 2), ('nsubj', 2, 1), ('nmod:assmod', 5, 3), ('compound:nn', 5, 4), ('nmod:assmod', 8, 5), ('case', 5, 6), ('dep', 8, 7), ('dobj', 2, 8), ('punct', 2, 9), ('conj', 2, 10), ('aux:asp', 10, 11), ('nummod', 16, 12), ('mark:clf', 12, 13), ('amod', 16, 14), ('case', 14, 15), ('dobj', 10, 16), ('punct', 2, 17), ('case', 29, 18), ('nmod:assmod', 21, 19), ('case', 19, 20), ('conj', 29, 21), ('cc', 29, 22), ('advmod', 24, 23), ('acl', 29, 24), ('nummod', 27, 25), ('mark:clf', 25, 26), ('dobj', 24, 27), ('mark', 24, 28), ('nmod:prep', 31, 29), ('punct', 31, 30), ('conj', 2, 31), ('nsubj', 34, 32), ('advmod', 34, 33), ('acl', 44, 34), ('aux:asp', 34, 35), ('dobj', 34, 36), ('punct', 34, 37), ('advmod', 39, 38), ('conj', 34, 39), ('advmod:rcomp', 39, 40), ('dobj', 39, 41), ('mark', 34, 42), ('compound:nn', 44, 43), ('dobj', 31, 44), ('punct', 2, 45)]
```
