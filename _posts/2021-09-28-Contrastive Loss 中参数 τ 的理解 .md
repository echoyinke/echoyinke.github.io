---
layout: post
title: "Contrastive Loss 中参数 τ 的理解"
subtitle: ''
author: "YiKe"
header-style: text
tags:
- impl
---

本文转自知乎[CVPR2021自监督学习论文: 理解对比损失的性质以及温度系数的作用](https://zhuanlan.zhihu.com/p/357071960)，在其基础上进行了部分删减以及公式化以达到更好地阅读效果

对比损失（Contrastive Loss）中的参数$\tau$是一个神秘的参数，大部分论文都默认采用较小的值来进行自监督对比学习（例如0.05），但是很少有文章详细讲解参数$\tau$的作用，本文将详解对比损失中的超参数$\tau$，并借此分析对比学习的核心机制。首先总结下本文的发现：

1.  对比损失是一个具备**困难负样本自发现**性质的损失函数，这一性质对于学习高质量的自监督表示是至关重要的。关注困难样本的作用是：对于那些已经远离的负样本，不需要让其继续远离，而主要聚焦在如何使没有远离的负样本远离，从而使得表示空间更均匀（Uniformity）
2.  $\tau$的作用是调节模型困难样本的关注程度：**$\tau$越小，模型越关注于将那些与本样本最相似的负样本分开**

#### 对比损失更关注困难负样本（Hardness-Awareness）

首先给出自监督学习广泛使用的对比损失（InfoNCE Loss）的形式：

$$ \mathcal{L}({x}_i) = -\log \left\[\frac{\exp (s_{i,i}/\tau)}{\sum_{k\neq i} \exp(s_{i,k}/\tau) + \exp (s_{i, i}/\tau)}\right\]\tag{1} $$

直观来说，该损失函数要求第$i$个样本和它另一个扩增的（正）样本之间的相似度$s_{i,i}$之间尽可能大，而与其它实例（负样本）之间的相似度$s_{i,k}$之间尽可能小。然而，其实还有很多损失函数可以满足这个要求，例如下面最简单的形式$\mathcal{L}_{\text{simple}}$：

$$ \mathcal{L}_{\text{simple}}({x}_i) = -s_{i,i} + \lambda \sum_{i\neq j}s_{i,j}\tag{2} $$

然而实际训练时，采用$\mathcal{L}_{\text{sample}}$作为损失函数效果非常不好，论文给出了使用式(1)和式(2)的性能对比（$\tau=0.07$）：

$$ \begin{array}{c|c|c} \hline \text{数据集} & \text{Contrastive Loss} & \text{Simple Loss}\\ \hline \text{CIFAR-10} & 79.75 & 74.83\\ \hline \text{CIFAR-100} & 51.82 & 39.31\\ \hline \text{ImageNet-100} & 71.53 & 48.09\\ \hline \text{SVHN} & 92.55 & 70.83 \\ \hline \end{array} $$

上面的结果显示，在所有数据集上Contrastive Loss都要远好于Simple Loss。作者通过研究发现，不同于Simple Loss，Contrastive Loss是一个困难样本自发现的损失函数。通过公式(2)可以发现，Simple Loss对所有的负样本给予了相同权重的惩罚$\frac{\partial \mathcal{L}_{\text{simple}}}{\partial s_{i,k}}=\lambda$。而Contrastive Loss则会自动给相似度更高的负样本比较高的惩罚，这一点可以通过对Contrastive Loss中不同负样本的相似度惩罚梯度观察得到：

$$ \text{对正样本的梯度：}\frac{\partial \mathcal{L}(x_i)}{\partial s_{i,i}}=-\frac{1}{\tau}\sum_{k\neq i} P_{i,k}\\ \text{对负样本的梯度：}\frac{\partial \mathcal{L}(x_i)}{\partial s_{i,j}}=\frac{1}{\tau}P_{i,j} $$

其中$P_{i,j}=\frac{\exp(s_{i,j/}\tau)}{\sum_{k\neq i} \exp(s_{i,k}/\tau) + \exp({s_{i,i}/\tau})}$。对于所有的负样本来说，$P_{i,j}$的分母项都是相同的，那么$s_{i,j}$越大，则负样本的梯度项$\frac{\partial \mathcal{L}(x_i)}{\partial s_{i,j}}=\frac{1}{\tau}P_{i,j}$也越大。也就是说，Contrastive Loss给予了更相似（困难）负样本更大的远离该样本的梯度。可以把不同的负样本想象成同极电荷在不同距离处的受力情况，距离越近的电荷受到的斥力越大，而越远的电荷受到的斥力越小。Contrastive Loss也是这样，这种性质有利于形成在超球面均匀分布的特性

#### 超参数$\tau$的作用

为了更具体的解释超参数$\tau$的作用，作者计算了两种极端情况，即$\tau$趋近于0和无穷大。当$\tau$趋近于0时：

$$ \begin{aligned} &\lim_{\tau \to 0^+}-\log \left\[\frac{\exp(s_{i,i}/\tau)}{\sum_{k\neq i} \exp{(s_{i,k}/\tau)}+\exp({s_{i,i}}/\tau)}\right\]\\ =&\lim_{\tau \to 0^+}\log \left\[1+ \sum_{k\neq i} \exp((s_{i,k} - s_{i,i})/\tau)\right\] \end{aligned}\tag{3} $$

简单点，我们仅考虑那些困难的负样本，即如果存在负样本$x_k$，有$\text{Sim}(x_i, x_k)\ge \text{Sim}(x_i,x_i^+)$，则称$x_k$为困难的负样本。此时式(3)可以改写为：

$$ \lim_{\tau \to 0^+}\log \left\[1+\sum_{\color{red}{s_{i,k} \ge s_{i,i}}}^k \exp((s_{i,k} - s_{i,i})/\tau)\right\]\tag{4} $$

因为$\tau \to 0^+$，我们直接省略常数1，并且将求和直接改为最大的$s_{i,k}$这一项，记作$s_{\text{max}}$，则式(4)可以改写为：

$$ \lim_{\tau \to 0^+} \frac{1}{\tau} \max\[s_{\text{max}} - s_{i,i},0\]\tag{5} $$

可以看出，此时Contrastive Loss退化为只关注最困难的负样本的损失函数。而当$\tau$趋近于无穷大时：

$$ \begin{aligned} &\lim_{\tau \to +\infty} -\log \left\[\frac{\exp({s_{i,i}}/\tau)}{\sum_{k\neq i}\exp(s_{i,k}/\tau) + \exp(s_{i,i}/\tau)}\right\]\\ =&\lim_{\tau \to +\infty} -\frac{1}{\tau}s_{i,i} + \log \sum_k \exp(s_{i,k}/\tau)\\ =& \lim_{\tau \to +\infty} -\frac{1}{\tau} s_{i,i} + \log \left\[N (1 + (\frac{1}{N}\sum_k \exp({s_{i,k}}/\tau)-1))\right\]\\ =&\lim_{\tau \to +\infty} -\frac{1}{\tau} s_{i,i} + \log \left\[1 + (\frac{1}{N}\sum_k \exp({s_{i,k}}/\tau)-1)\right\]+ \log N\\ =&\lim_{\tau \to +\infty} -\frac{1}{\tau}s_{i,i} + (\frac{1}{N}\sum_k \exp({s_{i,k}}/\tau)-1) + \log N\\ =&\lim_{\tau \to +\infty} -\frac{1}{\tau}s_{i,i} + \frac{1}{N\tau}\sum_{k} s_{i,k} + \log N\\ =&\lim_{\tau \to +\infty} \frac{1-N}{N\tau}s_{i,i} + \frac{1}{N\tau} \sum_{k\neq i} s_{i,k} + \log N \end{aligned}\tag{6} $$

> 上述等式推导利用了$\ln(1+x)$和$e^x$的泰勒展开，或者说等价无穷小

此时Contrastive Loss对所有负样本的权重都相同（$\frac{1}{N\tau}$），即Contrastive Loss失去了对困难样本关注的特性。有趣的是，当$\tau = \frac{N-1}{N}$时，对比损失$\mathcal{L}(x_i)$与前面提到的$\mathcal{L}_{\text{simple}}$几乎一样

论文作者通过上面两个极限情况分析了超参数$\tau$的作用：随着$\tau$的增大，Contrastive Loss倾向于“一视同仁”；随着$\tau$的减小，Contrastive Loss变得倾向于关注最困难的样本

#### References

+   [CVPR2021自监督学习论文: 理解对比损失的性质以及温度系数的作用](https://zhuanlan.zhihu.com/p/357071960)
+   [Understanding the Behaviour of Contrastive Loss](https://arxiv.org/pdf/2012.09740.pdf)

