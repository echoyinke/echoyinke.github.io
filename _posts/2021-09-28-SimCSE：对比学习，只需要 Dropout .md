---
layout: post
title: "SimCSE：对比学习，只需要 Dropout"
subtitle: ''
author: "YiKe"
header-style: text
tags:
- impl
---


要说2021年上半年NLP最火的论文，想必非[《SimCSE: Simple Contrastive Learning of Sentence Embeddings》](https://arxiv.org/abs/2104.08821)莫属。SimCSE的全称是**Simple C**ontrastive **S**entence **E**mbedding

#### Sentence Embedding

Sentence Embedding一直是NLP领域的一个热门问题，主要是因为其应用范围比较广泛，而且作为很多任务的基石。获取句向量的方法有很多，常见的有直接将\[CLS\]位置的输出当做句向量，或者是对所有单词的输出求和、求平均等。但以上方法均被证明存在各向异性（Anisotropy）问题。通俗来讲就是模型训练过程中会产生Word Embedding各维度表征不一致的问题，从而使得获得的句向量也无法直接比较

目前比较流行解决这一问题的方法有：

1.  线性变换：BERT-flow、BERT-Whitening。这两者更像是后处理，通过对BERT提取的句向量进行某些变换，从而缓解各向异性问题
2.  对比学习：SimCSE。 对比学习的思想是拉近相似的样本，推开不相似的样本，从而提升模型的句子表示能力

#### Unsupervised SimCSE

![](https://z3.ax1x.com/2021/07/21/WUYMss.png#shadow)

SimCSE利用自监督学习来提升句子的表示能力。由于SimCSE没有标签数据（无监督），所以把每个句子本身视为相似句子。说白了，本质上就是$(\text{自己},\text{自己})$作为正例、$(\text{自己},\text{别人})$作为负例来训练对比学习模型。当然，其实远没有这么简单，如果仅仅只是完全相同的两个样本作正例，那么泛化能力会大打折扣。一般来说，我们会使用一些数据扩增手段，让正例的两个样本有所差异，但是在NLP中如何做数据扩增本身也是一个问题，SimCSE提出了一个极为简单优雅的方案：**直接把Dropout当做数据扩增！**

具体来说，$N$个句子经过带Dropout的Encoder得到向量$\boldsymbol{h}_1^{(0)},\boldsymbol{h}_2^{(0)},...,\boldsymbol{h}_N^{(0)}$，然后让这批句子再重新过一遍Encoder（这时候是另一个随机Dropout）得到向量$\boldsymbol{h}_1^{(1)},\boldsymbol{h}_2^{(1)},...,\boldsymbol{h}_N^{(1)}$，我们可以将$(\boldsymbol{h}_i^{(0)},\boldsymbol{h}_i^{(1)})$视为一对（略有不同的）正例，那么训练目标为

$$ \ell_i=-\log \frac{e^{\text{sim}(\boldsymbol{h}_i^{(0)},\boldsymbol{h}_i^{(1)})/\tau}}{\sum_{j=1}^N e^{\text{sim}(\boldsymbol{h}_i^{(0)},\boldsymbol{h}_j^{(1)})/\tau}}\tag{1} $$

其中，$\text{sim}(\boldsymbol{h}_1, \boldsymbol{h}_2)=\frac{\boldsymbol{h}_1^T\boldsymbol{h}_2}{\Vert \boldsymbol{h}_1\Vert \cdot \Vert \boldsymbol{h}_2\Vert}$。实际上式(1)如果不看$-\log$和$\tau$的部分，剩下的部分非常像是$\text{Softmax}$。论文中设定$\tau = 0.05$，至于这个$\tau$有什么作用，我在网上看到一些解释：

1.  如果直接使用余弦相似度作为logits输入到$\text{Softmax}$，由于余弦相似度的值域是$\[-1,1\]$，范围太小导致$\text{Softmax}$无法对正负样本给出足够大的差距，最终结果就是模型训练不充分，因此需要进行修正，除以一个足够小的参数$\tau$将值进行放大
2.  超参数$\tau$会将模型更新的重点，聚焦到有难度的负例，并对它们做相应的惩罚，难度越大，也即是与$\boldsymbol{h}_i^{(0)}$距离越近，则分配到的惩罚越多。其实这也比较好理解，我们将$\text{sim}(\boldsymbol{h}_i^{(0)},\boldsymbol{h}_j^{(1)})$除以$\tau$相当于同比放大了负样本的logits值，如果$\tau$足够小，那么那些$\text{sim}(\boldsymbol{h}_i^{(0)},\boldsymbol{h}_j^{(1)})$越靠近1的负样本，经过$\tau$的放大后会占主导

> 个人觉得没有严格的数学证明，单从感性的角度去思考一个式子或者一个符号的意义是不够的，因此在查阅了一些资料后我将$\tau$这个超参数的作用整理成了另一篇文章：[Contrastive Loss中参数$\tau$的理解](https://wmathor.com/index.php/archives/1581/)

总结一下SimCSE的方法，个人感觉实在是太巧妙了，因为给两个句子让人类来判断是否相似，这其实非常主观，例如：“我喜欢北京”跟“我不喜欢北京”，请问这两句话到底相不相似？模型就像是一个刚出生的孩子，你教它这两个句子相似，那它就认为相似，你教它不相似，于是它以后见到类似的句子就认为不相似。此时，模型的性能或者准确度与训练过程、模型结构等都没有太大关系，真正影响模型预测结果的是人，或者说是人标注的数据

但是如果你问任何一个人“我喜欢北京”跟“我喜欢北京”这两句话相不相似，我想正常人没有说不相似的。SimCSE通过Dropout生成正样本的方法可以看作是数据扩增的最小形式，因为原句子和生成的句子语义是完全一致的，只是生成的Embedding不同而已。这样做避免了人为标注数据，或者说此时的样本非常客观

#### Alignment and Uniformity

对比学习的目标是从数据中学习到一个优质的语义表示空间，那么如何评价这个表示空间的质量呢？[Wang and Isola(2020)](https://arxiv.org/abs/2005.10242)提出了衡量对比学习质量的两个指标：alignment和uniformity，其中alignment计算$x_i$和$x_i^+$的平均距离：

$$ \ell_{\text{align}} \triangleq \mathop{\mathbb{E}}\limits_{(x, x^+)\sim p_{\text{pos}}} \Vert f(x) - f(x^+)\Vert^2\tag{2} $$

![](https://z3.ax1x.com/2021/07/21/WUYAdP.png#shadow)

而uniformity计算向量整体分布的均匀程度：

$$ \ell_{\text {uniform }} \triangleq \log \mathop{\mathbb{E}}\limits_{x, y \stackrel{i . i . d}{\sim} p_{\text{data}}} e^{-2\Vert f(x)-f(y)\Vert^{2}}\tag{3} $$

![](https://z3.ax1x.com/2021/07/21/WUYEIf.png#shadow)

我们希望这两个指标都尽可能低，也就是一方面希望正样本要挨得足够近，另一方面语义向量要尽可能地均匀分布在超球面上，因为均匀分布的信息熵最高，分布越均匀则信息保留的越多。作者从维基百科中随机抽取十万条句子来微调BERT，并在STS-B dev上进行测试，实验结果如下表所示：

其中None是作者提出的随机Dropout方法，其余方法均是在None的基础上对$x_{i}^+$进行改变，可以看到，追加显式数据扩增方法均会不同程度降低模型性能，效果最接近Dropout的是删除一个单词，但是删除一个单词并不能对uniformity带来很大的提升，作者也专门做了个实验来证明，如下图所示：

![](https://z3.ax1x.com/2021/07/21/WUYZi8.png#shadow)

#### Connection to Anisotropy

近几年不少研究都提到了语言模型生成的语义向量分布存在各向异性的问题，在探讨为什么Contrastive Learning可以解决词向量各向异性问题前，我们先来了解一下，什么叫各向异性。具体来说，假设我们的词向量设置为2维，如果各个维度上的基向量单位长度不一样，就是各向异性(Anisotropy)

例如下图中，基向量是非正交的，且各向异性（单位长度不相等），计算$x_1$与$x_2$的cos相似度为0，$x_1$与$x_3$的余弦相似度也为0。但是我们从几何角度上看，$x_1$与$x_3$其实是更相似的，可是从计算结果上看，$x_1$与$x_2$和$x_3$的相似度是相同的，这种不正常的原因即是各向异性造成的

![](https://z3.ax1x.com/2021/09/26/4yxgaj.png#shadow)

SimCSE的作者证明了当负样本数量趋于无穷大时，对比学习的训练目标可以渐近表示为：

$$ \begin{aligned} &-\frac{1}{\tau}\mathop{\mathbb{E}}\limits_{(x,x^+)\sim p_{\text{pos}}}\left\[f(x)^Tf(x^+)\right\]\\ &+\mathop{\mathbb{E}}\limits_{x\sim p_{\text{data}}} \left\[\log \mathop{\mathbb{E}}\limits_{x^-\sim p_{\text{data}}}\left\[e^{f(x)^Tf(x^-)/\tau}\right\] \right\] \end{aligned} \tag{4} $$

> 稍微解释一下这个式子，为了方便起见，接下来将$\frac{1}{\tau}\mathop{\mathbb{E}}\limits_{(x,x^+)\sim p_{\text{pos}}}\left\[f(x)^Tf(x^+)\right\]$称为第一项，$\mathop{\mathbb{E}}\limits_{x\sim p_{\text{data}}}\left\[\log \mathop{\mathbb{E}}\limits_{x^-\sim p_{\text{data}}}\left\[e^{f(x)^Tf(x^-)/\tau}\right\]\right\]$称为第二项。我们的最终目的是希望式(4)越小越好，具体来说，如果第一项越大、第二项越小，那么整体结果就非常小了。第一项大，则说明正样本对之间的相似度大；第二项小，则说明负样本对之间的相似度小，这也是我们最终希望看到的模型表现
>
> 接着我们尝试着从式(1)变换到式(4)，注意实际上$f(x)=\boldsymbol{h}_{i}^{(0)},f(x^+)=\boldsymbol{h}_{i}^{(1)},f(x^-)=\boldsymbol{h}_{j}^{(1)}$
>
> $$ \begin{aligned} \ell_i&=-\log \frac{e^{\text{sim}(\boldsymbol{h}_i^{(0)},\boldsymbol{h}_i^{(1)})/\tau}}{\sum_{j=1}^N e^{\text{sim}(\boldsymbol{h}_i^{(0)},\boldsymbol{h}_j^{(1)})/\tau}}\\ &=-\log (e^{\text{sim}(\boldsymbol{h}_i^{(0)},\boldsymbol{h}_i^{(1)})/\tau})+\log \sum_{j=1}^N e^{\text{sim}(\boldsymbol{h}_i^{(0)},\boldsymbol{h}_j^{(1)})/\tau}\\ &= {\color{red}{-\frac{1}{\tau}\text{sim}(\boldsymbol{h}_i^{(0)},\boldsymbol{h}_i^{(1)})}} + {\color{blue}{\log \sum_{j=1}^N e^{\text{sim}(\boldsymbol{h}_i^{(0)},\boldsymbol{h}_j^{(1)})/\tau}}} \end{aligned} $$
>
> 从下面开始，就不存在严格的等于了，而是一些等价或者正比关系。例如原本$\text{sim}(\boldsymbol{h}_1, \boldsymbol{h}_2)=\frac{\boldsymbol{h}_1^T\boldsymbol{h}_2}{\Vert \boldsymbol{h}_1\Vert \cdot \Vert \boldsymbol{h}_2\Vert}$，这里我们把分母省略掉，改成期望，同时将求和也改为期望，则
>
> $$ \begin{aligned} {\color{red}{-\frac{1}{\tau}\text{sim}(\boldsymbol{h}_i^{(0)},\boldsymbol{h}_i^{(1)})}} \propto-\frac{1}{\tau}\mathop{\mathbb{E}}\limits_{(x,x^+)\sim p_{\text{pos}}}\left\[f(x)^Tf(x^+)\right\]\\ {\color{blue}{\log \sum_{j=1}^N e^{\text{sim}(\boldsymbol{h}_i^{(0)},\boldsymbol{h}_j^{(1)})/\tau}}}\propto\mathop{\mathbb{E}}\limits_{x\sim p_{\text{data}}}\left\[\log \mathop{\mathbb{E}}\limits_{x^-\sim p_{\text{data}}}\left\[e^{f(x)^Tf(x^-)/\tau}\right\]\right\] \end{aligned} $$

我们可以借助Jensen不等式进一步推导第二项的下界：

$$ \begin{aligned} \mathop{\mathbb{E}}\limits_{x\sim p_{\text{data}}}\left\[\log \mathop{\mathbb{E}}\limits_{x^-\sim p_{\text{data}}}\left\[e^{f(x)^Tf(x^-)/\tau}\right\]\right\]&=\frac{1}{m}\sum_{i=1}^m\log (\frac{1}{m}\sum_{j=1}^m e^{\boldsymbol{h}_i^T \boldsymbol{h}_j / \tau})\\ &\ge \frac{1}{\tau m^2} \sum_{i=1}^m \sum_{j=1}^m\boldsymbol{h}_i^T \boldsymbol{h}_j \end{aligned}\tag{5} $$

> 首先等号的部分很容易理解，就是把期望改成了概率求和的形式，并且把$f(x)$和$f(x^-)$又改回$\boldsymbol{h}_i,\boldsymbol{h}_j$的形式。可能有同学不太了解Jensen不等式，这里我简单科普一下。对于一个凸函数$f(x)$，若$\lambda_i \ge 0$且$\sum_i \lambda_i=1$，则有
>
> $$ f(\sum_{i} \lambda_i x_i) \leq \sum_i \lambda_i f(x_i) $$
>
> 回到式(5)的证明，由于$\log$是凸函数，同时我们将$\frac{1}{m}$看作是$\lambda_i$，$e^{\boldsymbol{h}_i^T \boldsymbol{h}_j/\tau}$看作是$x_i$，应用Jensen不等式可得
>
> $$ \begin{aligned} \frac{1}{m}\sum_{i=1}^m\log (\frac{1}{m}\sum_{j=1}^m e^{\boldsymbol{h}_i^T \boldsymbol{h}_j / \tau})&\ge \frac{1}{m^2}\sum_{i=1}^m\sum_{j=1}^m \log e^{\boldsymbol{h}_i^T \boldsymbol{h}_j/\tau}\\ &= \frac{1}{\tau m^2} \sum_{i=1}^m \sum_{j=1}^m\boldsymbol{h}_i^T \boldsymbol{h}_j \end{aligned} $$

算了半天，回顾一下我们的终极目标是要优化式(4)，或者说最小化式(4)的第二项。设$ \mathbf{W}$为${x_i}_{i=1}^m$对应的**Sentence Embedding矩阵**，即$\mathbf{W}$的第$i$行是$\boldsymbol{h}_i$。那么此时优化第二项等价于最小化$\mathbf{W}\mathbf{W}^T$的上界。为什么？因为$\text{Sum}(\mathbf{W}\mathbf{W}^T)=\sum_{i=1}^m \sum_{j=1}^m\boldsymbol{h}_i^T \boldsymbol{h}_j$！假设我们已经标准化了$\boldsymbol{h}_i$，此时$\mathbf{W}\mathbf{W}^T$的对角线元素全为1，$ \text{tr}(\mathbf{W}\mathbf{W}^T) $为特征值之和，是一个常数。根据[Merikoski (1984)](https://linkinghub.elsevier.com/retrieve/pii/0024379584900788)的结论，如果$\mathbf{W}\mathbf{W}^T$的所有元素均为正值，则$\text{Sum}(\mathbf{W}\mathbf{W}^T)$是$\mathbf{W}\mathbf{W}^T$最大特征值的上界，因此，当我们最小化第二项时，其实是在间接最小化$\mathbf{W}\mathbf{W}^T$的最大特征值，也就是隐式地压平了嵌入空间的奇异谱，或者说使得嵌入空间的分布更均匀

到此为止，个人觉得已经将SimCSE核心内容讲的够清楚了，至于说原论文当中的监督学习部分，这里就不多赘言，因为本质上就是修改正样本对和负样本对的定义罢了

#### Results

原论文的实验非常丰富，读者可以仔细阅读原文，这里简单贴出一个实验对比图

![](https://z3.ax1x.com/2021/07/21/WwZ22q.png#shadow)

总体结果没有什么好分析的，通过上图我们可以知道SimCSE在多个数据集上都达到了SOTA，并且作者发现，在原有的训练目标的基础上加入MLM预训练目标，将两个目标的loss按比例相加$\ell + \lambda \ell^{mlm}$一起训练，能够防止SimCSE忘记token级别的知识，从而提升模型效果。这倒是让我感觉有点讶异的，做了那么多的操作，好不容易使得模型能够比较好的提取句子级别的特征了，结果token级别的知识又忘记了，真是拆了东墙补西墙

#### Code

虽然理论上我们说SimCSE是将同一个批次内的句子分别送入两个Encoder（这两个Encoder仅仅只是Dropout不同），但实现的时候我们其实是将一个batch内的所有样本复制一遍，然后通过一次Encoder即可。假设初始输入为$\[A, B\]$两条句子，首先复制一遍$\[A, A, B, B\]$，那么经过Encoder得到的句向量为$\[\boldsymbol{h}_A^{(0)}, \boldsymbol{h}_A^{(1)}, \boldsymbol{h}_B^{(0)}, \boldsymbol{h}_B^{(1)}\]$，现在的问题在于，我们的label是什么？

很明显我们知道如果给定$\boldsymbol{h}_A^{(0)},\boldsymbol{h}_A^{(1)}$，他们的label是1；如果给定$\boldsymbol{h}_B^{(0)},\boldsymbol{h}_B^{(1)}$，他们的label是1，其它情况均是0，所以给我们可以给出下面这样一张表格（标签为1的地方是相同句子不同Embedding的位置）

$$ \begin{array}{c|c|c|c|c} \hline & \boldsymbol{h}_A^{(0)} & \boldsymbol{h}_A^{(1)} & \boldsymbol{h}_B^{(0)} & \boldsymbol{h}_B^{(1)}\\ \hline \boldsymbol{h}_A^{(0)} & 0 & 1 & 0 & 0\\ \hline \boldsymbol{h}_A^{(1)} & 1 & 0 & 0 & 0\\ \hline \boldsymbol{h}_B^{(0)} & 0 & 0 & 0 & 1\\ \hline \boldsymbol{h}_B^{(1)} & 0 & 0 & 1 & 0\\ \hline \end{array} $$

上面的表格可以转换为label：$\[1, 0, 3, 2\]$。假设原始batch内有4条句子，复制后就共有8条句子，按照上述表格的排列方式，我们可以得到label：$\[1, 0, 3, 2, 5, 4, 7, 6\]$。按照这个规律生成label就可以了，而且这个规律其实挺明显的，就不解释了

```python
import torch
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def SimCSE_loss(pred, tau=0.05):
    ids = torch.arange(0, pred.shape[0], device=device)
    y_true = ids + 1 - ids % 2 * 2
    similarities = F.cosine_similarity(pred.unsqueeze(1), pred.unsqueeze(0), dim=2)

    # 屏蔽对角矩阵，即自身相等的loss
    similarities = similarities - torch.eye(pred.shape[0], device=device) * 1e12
    similarities = similarities / tau
    return torch.mean(F.cross_entropy(similarities, y_true))

pred = torch.tensor([[0.3, 0.2, 2.1, 3.1],
        [0.3, 0.2, 2.1, 3.1],
        [-1.79, -3, 2.11, 0.89],
        [-1.79, -3, 2.11, 0.89]])
SimCSE_loss(pred)
```

#### References

+   [SimCSE: Simple Contrastive Learning of Sentence Embeddings](https://arxiv.org/pdf/2104.08821.pdf)
+   [SimCSE: Simple Contrastive Learning of Sentence Embeddings（知乎）](https://zhuanlan.zhihu.com/p/368353121)
+   [中文任务还是SOTA吗？我们给SimCSE补充了一些实验](https://kexue.fm/archives/8348)
+   [SimCSE论文解读](https://zhuanlan.zhihu.com/p/369075953)
+   [SimCSE对比学习: 文本增广是什么牛马，我只需要简单Dropout两下](https://blog.csdn.net/weixin_45839693/article/details/116302914)
+   [张俊林：对比学习研究进展精要](https://mp.weixin.qq.com/s/xYlCAUIue_z14Or4oyaCCg)
+   [SimCSE论文超强解析](https://zhuanlan.zhihu.com/p/377612458)
+   [超细节的对比学习和SimCSE知识点](https://zhuanlan.zhihu.com/p/378340148)
+   [Bert中的词向量各向异性具体什么意思啊？](https://www.zhihu.com/question/460991118)