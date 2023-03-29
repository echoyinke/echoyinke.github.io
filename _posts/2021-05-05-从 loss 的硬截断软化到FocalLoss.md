---
layout: post
title: "从loss的硬截断软化到FocalLoss"
subtitle: ''
author: "YiKe"
header-style: text
tags:
- impl
---


对于二分类模型，我们总希望模型能够给正样本输出1，负样本输出0，但限于模型的拟合能力等问题，一般来说做不到这一点。而事实上在预测中，我们也是认为大于0.5的就是正样本了，小于0.5的就是负样本。这样就意味着，我们可以“有选择”地更新模型，**比如，设定一个阈值为0.6，那么模型对某个正样本的输出大于0.6，我就不根据这个样本来更新模型了，模型对某个负样本的输出小于0.4，我也不根据这个样本来更新模型了，只有在0.4~0.6之间的，才让模型更新，这时候模型会更“集中精力”去关心那些“模凌两可”的样本，从而使得分类效果更好**，这跟传统的SVM思想是一致的

不仅如此，这样的做法**理论上**还能防止过拟合，因为它防止了模型专门挑**那些容易拟合的样本**来"拼命"拟合（是的损失函数下降），这就好比老师只关心优生，希望优生能从80分提高到90分，而不想办法提高差生的成绩，这显然不是一个好老师

#### 修正的交叉熵损失（硬截断）

怎样才能达到我们上面说的目的呢？很简单，调整损失函数即可，这里主要借鉴了hinge loss和triplet loss的思想。一般常用的交叉熵损失函数是（$y$为one-hot表示，$\\hat{y}$为经过softmax后的输出）：

$$ L\_{old} = -\\sum y \\log \\hat{y} $$

> 实际上交叉熵损失函数的严格定义为：
>
> $$ L(\\hat{y}, y)=-\\sum (y\\log \\hat{y} + (1-y) \\log (1-\\hat{y})) $$
>
> 但PyTorch官方实现的`CrossEntropyLoss()`函数的形式是：
>
> $$ L(\\hat{y}, y) = -\\sum y\\log \\hat{y} $$
>
> 实际上究竟用哪个公式并不重要，这里提一下是为了避免读者误将PyTorch版交叉熵损失认为是原本的交叉熵损失的样貌

假设阈值选定为$m=0.6$，这个阈值原则上大于0.5均可。引入单位阶跃函数$\\theta(x)$：

$$ \\theta(x) = \\left\\{\\begin{aligned}&1, x > 0\\\\ &\\frac{1}{2}, x = 0\\\\ &0, x < 0\\end{aligned}\\right. $$

那么，考虑新的损失函数：

$$ L\_{new} = -\\sum\_y \\lambda(y, \\hat{y}) y\\log \\hat{y} $$

其中

$$ \\lambda(y, \\hat{y}) = 1-\\theta(y-m)\\theta(\\hat{y}-m)-\\theta(1-m-y)\\theta(1-m-\\hat{y}) $$

即

$$ \\lambda(y,\\hat{y})=\\left\\{\\begin{aligned}&0,\\,(y=1\\text{且}\\hat{y} > m)\\text{或}(y=0\\text{且}\\hat{y} < 1-m)\\\\ &1,\\,\\text{其他情形}\\end{aligned}\\right. $$

$L\_{new}$就是在交叉熵的基础上加入了修正项$\\lambda(y,\\hat{y})$，这一项意味着，当进入一个正样本时，那么$y=1$，显然

$$ \\lambda(1, \\hat{y})=1-\\theta(\\hat{y} - m) $$

这时候，如果$\\hat{y}>m$，那么$\\lambda(1, \\hat{y})=0$，这时交叉熵自动为0（达到最小值）；反之，$\\hat{y}<m$则有$\\lambda(1, \\hat{y})=1$，此时保持交叉熵，也就是说，**如果正样本的输出已经大于$m$了，那就不更新参数了，小于$m$才继续更新；类似地可以分析负样本的情形，如果负样本的输出已经小于$1-m$了，那就不更新参数了，大于$1-m$才继续更新**

这样一来，只要将原始的交叉熵损失，换成修正的交叉熵$L\_{new}$，就可以达到我们设计的目的了。下面是笔者利用PyTorch实现的多分类Loss（支持二分类），Keras版本请查看苏剑林大佬的[这篇博客](https://kexue.fm/archives/4293)

```python
import torch
import numpy as np
import torch.nn as nn
theta = lambda t: (torch.sign(t) + 1.) / 2.
sigmoid = lambda t: (torch.sigmoid(1e9 * t))

class loss(nn.Module):
  def __init__(self, theta, num_classes=2, reduction='mean', margin=0.5):
    super().__init__()
    self.theta = theta
    self.num_classes = num_classes
    self.reduction = reduction
    self.m = margin

  def forward(self, pred, y):
    '''
    pred: 2-D [batch, num_classes]. Softmaxed, no log
    y: 1-D [batch]. Index, but one-hot
    '''
    y_onehot = torch.tensor(np.eye(self.num_classes)[y]) # 2-D one-hot
    lambda_y_pred = 1 - self.theta(y_onehot - self.m) * \
                             self.theta(pred - self.m) \
                           - self.theta(1 - self.m - y_onehot) * \
                             self.theta(1 - self.m - pred)

    weight = torch.sign(torch.sum(lambda_y_pred, dim = 1)).unsqueeze(0)
    cel = y_onehot * torch.log(pred)# + (1 - y_onehot) * torch.log(1 - pred)
    if self.reduction == 'sum':
      return -torch.mean(torch.mm(weight, cel).squeeze(0))
    else:
      return -torch.sum(torch.mm(weight, cel).squeeze(0))
    
y_pred = torch.randn(3, 3)
y_pred_softmax = nn.Softmax(dim=1)(y_pred)
y_pred_softmax.clamp_(1e-8, 0.999999)
label = torch.tensor([0, 2, 2])
loss_fn = loss(theta, 3, reduction='mean', margin=0.6)

print(loss_fn(y_pred_softmax, label).item())
```

修正后的交叉熵损失看上去很好，同样的情况下在测试集上的表现确有提升，但是所需要的迭代次数会大大增加

原因是这样的：以正样本为例，**我只告诉模型正样本的预测值大于0.6就不更新了，却没有告诉它要"保持"大于0.6**，所以下一阶段，它的预测值很有可能变回小于0.6了，虽然在下一个回合它还能被更新，这样反复迭代，理论上可以达到目的，但是迭代次数会大大增加。所以，要想改进的话，重点是**除了告诉模型正样本的预测值大于0.6就不更新了，还要告诉模型当其大于0.6后继续保持**。（好比老师看到一个学生及格了就不管了，这显然是不行的。如果学生已经及格，那么应该要想办法让他保持目前这个状态甚至变得更好，而不是不管）

#### 软化Loss

硬截断会出现不足，关键在于因子$\\lambda(y, \\hat{y})$是不可导的，或者说我们认为它导数为0，因此这一项不会对梯度有任何帮助，从而我们不能从它这里得到合理的反馈（也就是模型不知道"保持"意味着什么）

解决这个问题的一个方法就是"软化"这个loss，**"软化"就是把一些本来不可导的函数用一些可导函数来近似**，数学角度应该叫"光滑化"。这样处理之后本来不可导的东西就可导

其实$\\lambda(y, \\hat{y})$中不可导的部分是$\\theta(x)$，因此我们只要"软化"$\\theta(x)$即可，而软化它再容易不过了，只需要利用sigmoid函数！我们有

$$ \\theta(x)=\\lim\_{K\\to +\\infty}\\sigma(Kx) $$

所以很显然，我们只需要将$\\theta(x)$替换为$\\sigma(Kx)$即可：

$$ \\begin{aligned} \\lambda(y, \\hat{y}) = 1&-\\sigma(K(y-m))\\sigma(K(\\hat{y}-m))\\\\&-\\sigma(K(1-m-y))\\sigma(K(1-m-\\hat{y})) \\end{aligned} $$

#### Focal Loss

由于Kaiming大神的Focal Loss一开始是基于图像的二分类问题所提出的，所以下面我们首先以二分类的损失函数为例，并且设$m=0.5$（为什么Kaiming大神不是NLPer......）

二分类问题的标准loss是交叉熵

$$ L\_{ce} = -y\\log \\hat{y} - (1-y)\\log(1-\\hat{y})=\\left\\{\\begin{aligned}&-\\log(\\hat{y}),\\,\\text{当}y=1\\\\ &-\\log(1-\\hat{y}),\\,\\text{当}y=0\\end{aligned}\\right. $$

其中$y\\in \\{0, 1\\}$是真实标签，$\\hat{y}$是预测值。当然，对于二分类函数我们几乎都是用sigmoid函数激活$\\hat{y}=\\sigma(x)$，所以相当于

$$ L\_{ce} = -y\\log \\sigma(x) - (1-y)\\log\\sigma(-x)=\\left\\{\\begin{aligned}&-\\log \\sigma(x),\\,\\text{当}y=1\\\\ &-\\log\\sigma(-x),\\,\\text{当}y=0\\end{aligned}\\right. $$

> $1-\\sigma(x)=\\sigma(-x)$

引入硬截断后的二分类loss形式为

$$ L^\* = \\lambda(y, \\hat{y})\\cdot L\_{ce} $$

其中

$$ \\begin{aligned} \\lambda(y, \\hat{y})&=\\left\\{\\begin{aligned}&1-\\theta(\\hat{y}-0.5),\\,\\text{当}y=1\\\\ &1-\\theta(0.5 - \\hat{y}),\\,\\text{当}y=0\\end{aligned}\\right.\\\\ &=\\left\\{\\begin{aligned}&\\theta(0.5-\\hat{y}),\\,\\text{当}y=1\\\\ &\\theta(\\hat{y} - 0.5),\\,\\text{当}y=0\\end{aligned}\\right. \\end{aligned} $$

实际上，它也等价于

$$ \\lambda(y, \\hat{y}) = \\left\\{\\begin{aligned}&\\theta(-x),\\,\\text{当}y=1\\\\ &\\theta(x),\\,\\text{当}y=0\\end{aligned}\\right. $$

> 注意这里我并没有说"等于"，而是"等价于"，因为$\\theta(0.5-\\hat{y})$表示$\\hat{y}>0.5$时取0，小于0.5时取1；而$\\theta(-x)$表示$x>0$时取0，小于0时取1。$\\hat{y}>0.5$和$\\hat{y}<0.5$分别刚好对应$x>0$和$x<0$

因为$\\theta(x)=\\lim\\limits\_{K\\to +\\infty}\\sigma(Kx)$，所以很显然有

$$ L^\* =\\left\\{\\begin{aligned}&-\\sigma(-Kx)\\log \\sigma(x),\\,\\text{当}y=1\\\\ &-\\sigma(Kx)\\log\\sigma(-x),\\,\\text{当}y=0\\end{aligned}\\right. $$

* * *

以上仅仅只是我们根据已知内容推导的二分类交叉熵损失，Kaiming大神的Focal Loss形式如下：

$$ L\_{fl}=\\left\\{\\begin{aligned}&-(1-\\hat{y})^{\\gamma}\\log \\hat{y},\\,\\text{当}y=1\\\\ &-\\hat{y}^{\\gamma}\\log (1-\\hat{y}),\\,\\text{当}y=0\\end{aligned}\\right. $$

带入$\\hat{y} = \\sigma(x)$则有

$$ L\_{fl}=\\left\\{\\begin{aligned}&-\\sigma^{\\gamma}(-x)\\log \\sigma(x),\\,\\text{当}y=1\\\\ &-\\sigma^{\\gamma}(x)\\log\\sigma(-x),\\,\\text{当}y=0\\end{aligned}\\right. $$

特别地，**如果$K$和$\\gamma$都取1，那么$L^\*=L\_{fl}$！**

事实上$K$和$\\gamma$的作用是一样的，都是为了调节权重曲线的陡度，只是调节的方式不太一样。注意$L^\*$或$L\_{fl}$实际上都已经包含了对不均衡样本的解决办法，或者说，类别不均衡本质上就是分类难度差异的体现。**比如负样本远比正样本多的话，模型肯定会倾向于数目多的负类（可以想像模型直接无脑全部预测为负类），这时负类的$\\hat{y}^{\\gamma}$或$\\sigma(Kx)$都很小，而正类的$(1- \\hat{y})^{\\gamma}$或$\\sigma(-Kx)$都很大，这时模型就会开始集中精力关注正样本**

当然，Kaiming大神还发现对$L\_{fl}$做个权重调整，结果会有微小提升

$$ L\_{fl}=\\left\\{\\begin{aligned}&-\\alpha(1-\\hat{y})^{\\gamma}\\log \\hat{y},\\,\\text{当}y=1\\\\ &-(1-\\alpha)\\hat{y}^{\\gamma}\\log (1-\\hat{y}),\\,\\text{当}y=0\\end{aligned}\\right. $$

通过一系列调参，得到$\\alpha=0.25, \\gamma=2$（在他的模型上）的效果最好。注意在他的任务中，正样本是少数样本，也就是说，本来正样本难以“匹敌”负样本，但经过$(1−\\hat{y})^{\\gamma}$和$\\hat{y}^{\\gamma}$的"操控"后，也许形势还逆转了，因此要对正样本降权。不过我认为这样调整只是经验结果，理论上很难有一个指导方案来决定$\\alpha$的值，如果没有大算力调参，倒不如直接让$\\alpha=0.5$（均等）

#### 多分类

Focal Loss在多分类中的形式也很容易得到，其实就是

$$ L\_{fl} = -(1-\\hat{y})^\\gamma\\log\\hat{y\_t} $$

其中，$\\hat{y\_t}$是目标的预测值，一般是经过Softmax后的结果

#### 为什么Focal Loss有效？

这一节我们试着理解为什么Focal Loss有效，下图展示了不同$\\gamma$值下Focal Loss曲线。特别地，当$\\gamma=0$时，其形式就是CrossEntropy Loss

![](https://z3.ax1x.com/2021/05/05/gKA4ds.png#shadow)

在上图中，"蓝色"线表示交叉熵损失，X轴表示预测真实值的概率，Y轴是给定预测值下的损失值。从图像中可以看出，当模型以0.6的概率预测真实值时，交叉熵损失仍在0.5左右。因此为了减少损失，我们要求模型必须以更高的概率预测真实值。换句话说，交叉熵损失要求模型对真实值的预测结果非常有信心，但这反过来实际上会对性能产生负面影响

> 模型实际上可能变得过于自信（或者说过拟合），因此该模型无法更好的推广（鲁棒性不强）

Focal Loss不同于上述方案，从图中可以看出，使用$\\gamma >1$的Focal Loss可以减少模型预测正确概率大于0.5的损失。因此，在类别不平衡的情况下，Focal Loss会将模型的注意力转向稀有类别。实际上仔细观察上图我们还能分析得到：**更大的$\\gamma$值对模型预测概率的"宽容度"越高**。如何理解这句话？我们对比$\\gamma=2$和$\\gamma=5$的两条曲线，$\\gamma=5$时，模型预测概率只要大于0.3，Loss就非常小了；$\\gamma=2$时，模型预测概率至少要大于0.5，Loss才非常小，所以这变相是在人为规定置信度

下面是基于PyTorch实现支持多分类的Focal Loss代码，源自[https://github.com/yatengLG/Focal-Loss-Pytorch](https://github.com/yatengLG/Focal-Loss-Pytorch)，由于代码年久失修，有些在issue中提出的bug作者还没来得及修改，这里我贴出的代码是经过修改后的，其中需要注意的是$\\alpha$这个参数，**样本较多的类别应该分配一个较大的权重，而样本数较少的类别应该分配一个较小的权重**。这里我默认$\\alpha=0.75$相当于默认多分类中，第0个类别样本数比较大，如果举个具体的例子，在NER任务中，`other`这个这个类别对应的索引是0，而且`other`这个类别一般来说都特别多（大部分情况下是最多的），所以`other`分配到的权重应该是$\\alpha=0.75$，而其他类别的权重均为$1-\\alpha=0.25$

```python
import torch
from torch import nn
from torch.nn import functional as F

class focal_loss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes=3, size_average=True):
        """
        focal_loss损失函数, $-\alpha (1-\hat{y})^{\gamma} * CrossEntropyLoss(\hat{y}, y)$
        alpha: 类别权重. 当α是列表时, 为各类别权重, 当α为常数时, 类别权重为[α, 1-α, 1-α, ....]
        gamma: 难易样本调节参数.
        num_classes: 类别数量
        size_average: 损失计算方式, 默认取均值
        """
        super(focal_loss,self).__init__()
        self.size_average = size_average
        if isinstance(alpha, list):
            assert len(alpha) == num_classes   # α可以以list方式输入, 用于对不同类别精细地赋予权重
            print(" --- Focal_loss alpha = {}, 将对每一类权重进行精细化赋值 --- ".format(alpha))
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha < 1
            print(" --- Focal_loss alpha = {} --- ".format(alpha))
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1-alpha) # α 最终为[α, 1-α, 1-α, 1-α, 1-α, ...] size:[num_classes]

        self.gamma = gamma

    def forward(self, preds, labels):
        """
        preds: 预测类别. size:[B, C] or [B, S, C] B 批次, S长度, C类别数
        labels: 实际类别. size:[B] or [B, S] B批次, S长度
        """
        # assert preds.dim() == 2 and labels.dim()==1
        labels = labels.view(-1, 1) # [B * S, 1]
        preds = preds.view(-1, preds.size(-1)) # [B * S, C]
        self.alpha = self.alpha.to(preds.device)
        preds_logsoft = F.log_softmax(preds, dim=1) # 先softmax, 然后取log
        preds_softmax = torch.exp(preds_logsoft)    # softmax

        print(labels)
        print(preds_softmax)
        preds_softmax = preds_softmax.gather(1, labels)   # 这部分实现nll_loss ( crossempty = log_softmax + nll )
        preds_logsoft = preds_logsoft.gather(1, labels)
        alpha = self.alpha.gather(0, labels.view(-1))
        loss = -torch.mul(torch.pow((1-preds_softmax), self.gamma), preds_logsoft)  # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ

        loss = torch.mul(alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss
      
y_pred = torch.randn(3, 4, 5) # 3个样本, 5类
label = torch.tensor([[0, 3, 4, 1], [2, 1, 0, 0], [0, 0, 4, 1]]) # 0类特别多，应该给0类一个较大的$\alpha$权重
loss_fn = focal_loss(alpha = 0.75, num_classes=5)

print(loss_fn(y_pred, label).item())
```

### Focal Loss使用问题总结

在笔者研究Focal Loss的这几天，看了不少文章，其中也提到很多关于Focal Loss的问题，这里一一列出进行记录

#### 关于参数$\\alpha$（非常重要，仔细阅读）

很多读者可能想当然认为应该给样本较少的类别赋予一个较大的权重，实际上如果没有$(1-\\hat{y}^\\gamma)$以及$\\hat{y}^{\\gamma}$这两项，这么做确实没问题。但由于引入了这两项，本来样本少的类别应该是难分类的，结果随着模型的训练，样本多的类别变得难分类了，在这种情况下，我们应该给样本少的类别一个较小的权重，而给样本多的类别一个较大的权重

简单来说，添加$(1-\\hat{y}^\\gamma)$以及$\\hat{y}^{\\gamma}$是为了平衡正负样本，而添加$\\alpha$和$(1-\\alpha)$又是为了平衡$(1-\\hat{y}^\\gamma)$以及$\\hat{y}^{\\gamma}$，有一种套娃的思想在里面，平衡来平衡去

#### 训练过程

建议一开始训练不要使用Focal Loss。对于一般的分类问题，开始时先使用正常的CrossEntropyLoss，训练一段时间，确保网络的权重更新到一定的时候再更换为Focal Loss

#### 初始化参数

有一个非常小的细节，对于分类问题，我们一般会在最后通过一个Linear层，而这个Linear层的bias设置是有讲究的，一般初始化设为

$$ b = -\\log \\frac{1-\\pi}{\\pi} $$

其中，假设二分类中样本数少的类别共有$m$个，样本数多的类别共有$n$个（$m+n$等于总样本数），则$\\pi=\\frac{m}{m+n}$，为什么这样设计？

首先我们知道最后一层的激活函数是$\\sigma:\\frac{1}{1+e^{-(wx+b)}}$，因为默认初始化的情况下$w,b$均为0，此时不管你提取到的特征是什么，或者说不管你输入的是什么，经过激活之后的输出都是0.5（正类和负类都是0.5），这会带来什么问题？

假设我们使用二分类的CrossEntropyLoss

$$ L = -\\log (p) - (1 - y)\\log (p) $$

那么刚开始的时候，不管输入的是正样本还是负样本（假设负样本特别多），他们的误差都是$-\\log (0.5)$，而负样本的个数多得多，这么看，刚开始训练的时候，loss肯定要被负样本的误差带偏（模型会想方设法尽力全部预测成负样本，以降低loss）

但是如果我们对最后一层的bias使用上面的初始化呢？把$b$带入到$\\sigma$中

$$ \\frac{1}{1+e^{\\log(\\frac{1-\\pi}{\\pi})}}=\\frac{1}{1+(\\frac{1-\\pi}{\\pi})}=\\pi $$

对于正样本来说，$L=-\\log (\\pi)$；对于负样本来说，$L=-\\log (1-\\pi)$。由于$0<\\pi<1-\\pi<1$，所以$\\log(1-\\pi) < \\log(\\pi)$，这样做了以后，虽然可能所有负样本联合起来的损失仍然比正样本大，但相较于不初始化bias的情况要好很多

实际上我本人写代码的时候，尤其在`nn.Linear`中喜欢设置`bias=False`，即不添加bias，因为我认为`nn.Linear`多数情况下只是为了转换一下维度，进行一个线性变换的操作，所以加上bias可能会使得原本特征矩阵内的值变得怪怪的，但是这里最好还是加上

#### References

+   [文本情感分类（四）：更好的损失函数](https://kexue.fm/archives/4293)
+   [从loss的硬截断、软化到focal loss](https://kexue.fm/archives/4733)
+   [What is Focal Loss and when should you use it?](https://amaarora.github.io/2020/06/29/FocalLoss.html#so-why-did-that-work-what-did-focal-loss-do-to-make-it-work)
+   [focal loss理解与初始化偏置b设置解释](https://zhuanlan.zhihu.com/p/63626711)
+   [使用focal loss训练数据不平衡的模型](https://zhuanlan.zhihu.com/p/258506276))