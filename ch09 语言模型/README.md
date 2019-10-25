<h1>
第九章 语言模型
</h1>

<!-- TOC -->

- [1. 语言模型（Language Model，LM）的任务](#1-语言模型language-modellm的任务)
- [2. 语言模型评估：困惑度(Perplexity](#2-语言模型评估困惑度perplexity)
- [3. 语言模型的传统方法](#3-语言模型的传统方法)
- [4. 神经语言模型（NNLM）](#4-神经语言模型nnlm)
- [5. 使用语言模型进行生成](#5-使用语言模型进行生成)

<!-- /TOC -->

# 1. 语言模型（Language Model，LM）的任务
语言模型的任务是预测每个句子在语言中出现的概率。

正式来讲，语言模型就是给任何词序列 $w_{1:n}$ 分配一个概率， 也就是 $P\left(w_{1:n}\right)$。通过概
率的链式法则可以写成如下形式：**(n-gram)**

$$
P\left(w_{1:n}\right)=P\left(w_{1}\right) P\left(w_{2} | w_{1}\right) P\left(w_{3} | w_{1 : 2}\right) P\left(w_{4} | w_{1:3}\right) \cdots P\left(w_{n} | w_{1 : n-1}\right)
$$

使用**马尔可夫假设（ Markov-Assumption）**，该假设规定未来的状态和现在给定的状态是无
关的。 一个k阶马尔可夫假设假设序列中下一个词只依赖于其前k个词：
$$
P\left(w_{i+1} | w_{1:i}\right) \approx P\left(w_{i+1} | w_{i-k: i}\right)
$$
句子的概率估计就变成了
$$
P\left(w_{1 : n}\right) \approx \prod_{i=1}^{n} P\left(w_{i} | w_{i-k: i-1}\right)
$$

我们接下来的任务就是根据给定的大量文本准确地估算出$P\left(w_{i+1} | w_{i-k:i}\right)$

# 2. 语言模型评估：困惑度(Perplexity

困惑度度量是一个评价语言模型质量的良好指标，越低越好

$$
2^{-\frac{1}{n} \sum_{i=1}^{n} \log _{2} \operatorname{LM}\left(w_{i} | w_{1 : i-1}\right)}
$$

# 3. 语言模型的传统方法

$\widehat{p}\left(w_{i+1}=m | w_{i-k_{i} i}\right)$的最大似然估计是：
$$
\hat{p}_{\mathrm{MLE}}\left(w_{i+1}=m | w_{i-k : i}\right)=\frac{\#\left(w_{i-k : i+1}\right)}{\#\left(w_{i-k : i}\right)}
$$

- 一般的语料库不可能直接计算出某个句子出现的概率
  - 例：一个只考虑前 2 个词的三元文法语言模型， 10000 词的词表（非常小）。将有 10000^3 = 10^12 种可能的序列
- 解决方法1：**添加（add-α）平滑**技术，.它假设每个事件除了语料中观测的情况外，至少还发生 α 次 。 这个估计就变成了：
$$
\hat{p}_{\mathrm{add-\alpha}}\left(w_{i+1}=m | w_{i-k : i}\right)=\frac{\#\left(w_{i-k : i+1}\right)+\alpha}{\#\left(w_{i-k : i}\right)+\alpha|V|}
$$
$|V|$是词表大小，$0<\alpha \leqslant 1$

- 解决方法2：**退避 (back off)**，如果没有观测到 k 元文法，那么就基于（k-1) 元文法计算一个估计值
  - 例：**贾里尼克插值平滑**  (Jelinek Mercer interpolated smoothing)

- **解决方法3**： **Knerser Ney 平滑技术**(当前最佳的非神经网络语言模型技术)

>公式比较复杂且与主题无关，不作展开

# 4. 神经语言模型（NNLM）

![](https://img-blog.csdn.net/20160922200454004)

- 输入：n-1个之前的word（用词典库V中的index表示）
- 映射：通过|V|*D的矩阵C映射到D维
- 隐层：映射层连接大小为H的隐层
- 输出：输出层大小为|V|，表示|V|个词中每个词的概率

1. **网络第一层**：将 $C(w_{t-n+1}), … , C(w_{t-2})C(w_{t-1})$ 这n-1个向量拼接起来，形成一个(n-1)*m维的矩阵，记为矩阵$C$。
2. **网络第二层**：就是神经网络的隐藏层，直接使用 $d+HC$ 计算得到，H是权重矩阵，d是一个偏置向量。然后使用tanh作为激活函数。
3. **网络第三层**：输出一共有|V|个节点，每个节点 $y_i$ 表示下一个词为i的未归一化log概率，最后使用softmax激活函数将输出值y归一化成概率。y的计算公式：

$$
y=b+W C+U \tanh (d+H C)
$$
式中U是一个 $|V| \times h$ 的矩阵，表示隐藏层到输出层的参数；W是$|V| \times (n-1)m$的矩阵，这个矩阵包含了从输入层到输出层的直连边，就是从输入层直接到输出层的一个线性变换。
$$
p=\operatorname{softmax}(y)
$$

# 5. 使用语言模型进行生成

>尚未完工
