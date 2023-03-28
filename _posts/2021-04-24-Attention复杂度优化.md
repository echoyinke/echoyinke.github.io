---
layout: post
title: "Attention复杂度"
subtitle: 'Attention'
author: "YiKe"
header-style: text
tags:
- impl
---


众所周知，尽管基于Attention机制的Transformer类模型有着良好的并行性能，但它的空间和时间复杂度都是$\\mathcal{O}(n^2)$级别的，$n$是序列长度，所以当$n$比较大时Transformer模型的计算量难以承受。近来，也有不少工作致力于降低Transformer模型的计算量，比如模型剪枝、量化、蒸馏等精简技术，又或者修改Attention结构，使得其复杂度能降低到$\\mathcal{O}(n\\log⁡n)$甚至$\\mathcal{O}(n)$

论文[《Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention》](https://arxiv.org/abs/2006.16236)当中提到一种线性化Attention（Linear Attention）的方法，由此引发了我的兴趣，继而阅读了一些相关博客，有一些不错的收获，最后将自己对线性化Attention的理解汇总在此文中

#### Attention

当前最流行的Attention机制当属[Scaled-Dot Attention](https://arxiv.org/abs/1706.03762)，即

$$ \\begin{equation}\\text{Attention}(\\boldsymbol{Q},\\boldsymbol{K},\\boldsymbol{V}) = \\text{Softmax}\\left(\\boldsymbol{Q}\\boldsymbol{K}^{\\top}\\right)\\boldsymbol{V}\\tag{1}\\end{equation} $$

这里的$\\boldsymbol{Q}\\in \\mathbb{R}^{n\\times d\_k}, \\boldsymbol{K}\\in \\mathbb{R}^{m\\times d\_k}, \\boldsymbol{V}\\in \\mathbb{R}^{m\\times d\_v}$，简单起见我就没显式的写出Attention的缩放因子$\\frac{1}{\\sqrt{d}}$了。本文我们主要关心Self Attention的场景，所以为了介绍上的方便，统一设$\\boldsymbol{Q},\\boldsymbol{K},\\boldsymbol{V}\\in \\mathbb{R}^{n\\times d}$

#### 摘掉Softmax

读者也许想不到，制约Attention性能的关键因素，其实是定义里边的Softmax！事实上，简单地推导一下就可以得到这个结论。$QK^T$这一步我们得到一个$n\\times n$的矩阵，之后还要做一个Softmax

> 对一个$1\\times n$的行向量进行Softmax，时间复杂度是$O(n)$，但是对一个$n\\times n$矩阵的每一行做一个Softmax，时间复杂度就是$O(n^2)$

如果没有Softmax，那么Attention的公式就变为三个矩阵连乘$\\boldsymbol{QK^{\\top}V}$，而矩阵乘法是满足结合率的，所以我们可以先算$\\boldsymbol{K^{\\top}V}$，得到一个$d\\times d$的矩阵（这一步的时间复杂度是$O(d^2n)$），然后再用$Q$左乘它（这一步的时间复杂度是$O(d^2n)$），由于$d \\ll n$，所以这样算大致的时间复杂度只是$O(n)$

> 对于BERT base来说，$d=64$而不是768，why？因为768实际上是通过Multi-Head拼接得到的，而每个head的$d=64$

也就是说，去掉Softmax的Attention复杂度可以降到最理想的线性级别$\\mathcal{O}(n)$！这显然就是我们的终极追求：Linear Attention

#### 一般的定义

问题是，直接去掉Softmax还能算是Attention吗？他还能有标准的Attention的效果吗？为了回答这个问题，我们先将Scaled-Dot Attention的定义等价的改写为（本文的向量都是列向量）

$$ \\begin{equation}\\text{Attention}(\\boldsymbol{Q},\\boldsymbol{K},\\boldsymbol{V})\_i = \\frac{\\sum\\limits\_{j=1}^n e^{\\boldsymbol{q}\_i^{\\top}\\boldsymbol{k}\_j}\\boldsymbol{v}\_j}{\\sum\\limits\_{j=1}^n e^{\\boldsymbol{q}\_i^{\\top}\\boldsymbol{k}\_j}}\\tag{2}\\end{equation} $$

> 这里稍微解释下，首先我们知道$\\boldsymbol{Q},\\boldsymbol{K}\\in \\mathbb{R}^{n\\times d}$，令$\\boldsymbol{M} = \\boldsymbol{Q}\\times \\boldsymbol{K^{\\top}}$，由矩阵乘法法则可知，$\\boldsymbol{M}$的第一行是由$\\boldsymbol{Q}$的第一行乘以$\\boldsymbol{K^{\\top}}$的所有列得到的
>
> $\\text{Attention}(\\boldsymbol{Q},\\boldsymbol{K},\\boldsymbol{V})\_i$表示最终输出结果矩阵的第$i$行
>
> $\\boldsymbol{q}\_i^{\\top}$表示$\\boldsymbol{Q}\\in \\mathbb{R}^{n\\times d}$矩阵的第$i$行（行向量）
>
> $\\boldsymbol{k}\_j$表示$\\boldsymbol{K^{\\top}}\\in \\mathbb{R}^{d\\times n}$矩阵的第$j$列（列向量）
>
> $\\boldsymbol{v}\_j$表示$V^{\\top}\\in \\mathbb{R}^{d\\times n}$矩阵的的第$j$列（列向量）

所以，Scaled-Dot Attention其实就是以$e^{\\boldsymbol{q}\_i^{\\top}\\boldsymbol{k}\_j}$为权重对$\\boldsymbol{v}\_j$做加权平均。所以我们可以提出一个Attention的一般化定义

$$ \\begin{equation}\\text{Attention}(\\boldsymbol{Q},\\boldsymbol{K},\\boldsymbol{V})\_i = \\frac{\\sum\\limits\_{j=1}^n \\text{sim}(\\boldsymbol{q}\_i, \\boldsymbol{k}\_j)\\boldsymbol{v}\_j}{\\sum\\limits\_{j=1}^n \\text{sim}(\\boldsymbol{q}\_i, \\boldsymbol{k}\_j)}\\tag{3}\\end{equation} $$

也就是把$e^{\\boldsymbol{q}\_i^{\\top}\\boldsymbol{k}\_j}$换成$\\boldsymbol{q}\_i,\\boldsymbol{k}\_i$的一般函数$\\text{sim}(\\boldsymbol{q}\_i,\\boldsymbol{k}\_j)$，为了保留Attention相似的分布特性，我们要求$\\text{sim}(\\boldsymbol{q}\_i, \\boldsymbol{k}\_j)\\geq 0$恒成立。也就是说，我们如果要定义新的Attention，必须要保留式(3)的形式，并且满足$\\text{sim}(\\boldsymbol{q}\_i, \\boldsymbol{k}\_j)\\geq 0$

这种一般形式的Attention在CV中也被称为Non-Local网络，出自论文[《Non-local Neural Networks》](https://arxiv.org/abs/1711.07971v3)

#### 几个例子

如果直接去掉Softmax，那么就是$\\text{sim}(\\boldsymbol{q}\_i, \\boldsymbol{k}\_j) = \\boldsymbol{q}\_i^{\\top}\\boldsymbol{k}\_j$，问题是内积无法保证非负性，所以这还不是一个合理的选择。下面我们介绍几种可取的方案

值得一提的是，下面介绍的这几种Linear Attention，前两种来自CV领域，第三种是[苏剑林](https://www.spaces.ac.cn/archives/7546)大佬构思的（除了下面的介绍外，还有[EMANet](https://arxiv.org/abs/1907.13426)等CV领域对Attention的改进工作）

#### 核函数形式

一个自然的想法是：如果$\\boldsymbol{q}\_i, \\boldsymbol{k}\_j$的每个元素都是非负的，那么内积自然也是非负的。为了完成这点，我们可以给$\\boldsymbol{q}\_i, \\boldsymbol{k}\_j$各自加个激活函数$\\phi,\\varphi$，即

$$ \\begin{equation}\\text{sim}(\\boldsymbol{q}\_i, \\boldsymbol{k}\_j) = \\phi(\\boldsymbol{q}\_i)^{\\top} \\varphi(\\boldsymbol{k}\_j)\\tag{4}\\end{equation} $$

其中$\\phi(\\cdot), \\varphi(\\cdot)$是值域非负的激活函数。本文开头提到的论文[《Transformers are RNNs》](https://arxiv.org/abs/2006.16236)选择的是$\\phi(x)=\\varphi(x)=\\text{elu}(x)+1$，其中

$$ \\text{elu}(x)=\\begin{cases}x& \\text{if} \\ x>0\\\\ \\alpha (e^x-1) & \\text{if}\\ x<0\\end{cases} $$

> 常见的$\\alpha$取值为$\[0.1, 0.3\]$

非要讲故事的话，式(4)可以联想到"核方法"，尤其是$\\phi=\\varphi$时，$\\phi$就相当于一个核函数，而$\\langle \\phi(\\boldsymbol{q}\_i), \\phi(\\boldsymbol{k}\_j)\\rangle$就是通过核函数所定义的内积。这方面的思考可以参考论文[《Transformer dissection: An unified understanding for transformer’s attention via the lens of kernel》](https://arxiv.org/abs/1908.11775)，此处不做过多延伸

#### 妙用Softmax

另一篇更早的文章[《Efficient Attention: Attention with Linear Complexities》](https://arxiv.org/abs/1812.01243)则给出了一个更有意思的选择。它留意到在$\\boldsymbol{QK^{\\top}}$中，$\\boldsymbol{Q},\\boldsymbol{K}\\in \\mathbb{R}^{n\\times d}$，如果“$\\boldsymbol{Q}$在$d$那一维是归一化的，并且$\\boldsymbol{K}$在$n$那一维是归一化的”，那么$\\boldsymbol{QK^{\\top}}$就是自动满足归一化了，所以它给出的选择是

$$ \\begin{equation}\\text{Attention}(\\boldsymbol{Q},\\boldsymbol{K},\\boldsymbol{V}) = \\text{Softmax}\_2\\left(\\boldsymbol{Q}\\right)\\text{Softmax}\_1(\\boldsymbol{K})^{\\top}\\boldsymbol{V}\\tag{5}\\end{equation} $$

其中$\\text{Softmax}\_1$、$\\text{Softmax}\_2$分别表示在第一个$(n)$、第二个维度$(d)$进行Softmax运算。也就是说，这时候我们是各自给$\\boldsymbol{Q},\\boldsymbol{K}$加Softmax，而不是算完$\\boldsymbol{QK^{\\top}}$之后再加Softmax

其实可以证明这个形式也是式(4)​的一个特例，此时对应于$\\phi(\\boldsymbol{q}\_i)=\\text{Softmax}(\\boldsymbol{q}\_i),\\varphi(\\boldsymbol{k}\_j)=e^{\\boldsymbol{k}\_j}$，读者可以自行推导一下

#### 苏神的构思

在这里，苏神给出了一种构思。这个构思的出发点不再是式(4)，而是源于我们对原始定义(2)​的泰勒展开。由泰勒展开我们有

$$ \\begin{equation}e^{\\boldsymbol{q}\_i^{\\top}\\boldsymbol{k}\_j} \\approx 1 + \\boldsymbol{q}\_i^{\\top}\\boldsymbol{k}\_j\\tag{6}\\end{equation} $$

如果$\\boldsymbol{q}\_i^{\\top}\\boldsymbol{k}\_j\\geq -1$，那么就可以保证右端的非负性，从而可以让$\\text{sim}(\\boldsymbol{q}\_i, \\boldsymbol{k}\_j)=1 + \\boldsymbol{q}\_i^{\\top}\\boldsymbol{k}\_j$。到这里读者可能已经想到了，想要保证$\\boldsymbol{q}\_i^{\\top}\\boldsymbol{k}\_j\\geq -1$，只需要分别对$\\boldsymbol{q}\_i,\\boldsymbol{k}\_j$做$l\_2$归一化。所以，苏神最终提出的方案就是：

$$ \\begin{equation}\\text{sim}(\\boldsymbol{q}\_i, \\boldsymbol{k}\_j) = 1 + \\left( \\frac{\\boldsymbol{q}\_i}{\\Vert \\boldsymbol{q}\_i\\Vert}\\right)^{\\top}\\left(\\frac{\\boldsymbol{k}\_j}{\\Vert \\boldsymbol{k}\_j\\Vert}\\right)\\tag{7}\\end{equation} $$

> 若$\\boldsymbol{x}=\[x\_1,x\_2,...,x\_n\]$，则$\\Vert x\\Vert=\\sqrt{x\_1^2+x\_2^2+···+x\_n^2}$

这不同于式(4)，但理论上它更加接近原始的Scaled-Dot Attention

#### 实现

这里主要是针对苏神所提出的方法进行实现，但是由于笔者本人水平有限，因此最终实现的代码当中其实存在一些问题，主要是：

1.  从测试结果来看，改进后的计算速度并没有提升
2.  无法做到求和为1


```python
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        Q = F.normalize(Q, dim=3)
        K = F.normalize(K, dim=3)
        M = (torch.ones(Q.shape[0], Q.shape[1], Q.shape[2], K.shape[2]) + torch.matmul(Q, K.transpose(-1, -2))) # scores : [batch_size, n_heads, seq_len, seq_len]
        M_sum = torch.sum(M, dim=3)
        M = M / M_sum.unsqueeze(3).repeat(1, 1, 1, M.shape[3])
        attn = M.masked_fill(attn_mask, 0) # Fills elements of self tensor with value where mask is one.
        context = torch.matmul(attn, V)
        return context
```

如果您有更好的实现方法，还望不吝赐教

#### Reference

+   [线性Attention的探索：Attention必须有个Softmax吗？](https://www.spaces.ac.cn/archives/7546)