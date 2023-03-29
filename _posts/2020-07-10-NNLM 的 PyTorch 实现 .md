---
layout: post
title: "NNLM 的 PyTorch 实现"
subtitle: ''
author: "YiKe"
header-style: text
tags:
- impl
---


本文主要首先介绍一篇年代久远但意义重大的论文[A Neural Probabilistic Language Model(2003)](http://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)，然后给出PyTorch实现

### A Neural Probabilistic Language Model

本文算是训练语言模型的经典之作，Bengio将神经网络引入语言模型的训练中，并得到了词向量这个副产物。词向量对后面深度学习在自然语言处理方面有很大的贡献，也是获取词的语义特征的有效方法

其主要架构为三层神经网络，如下图所示

![](https://z3.ax1x.com/2021/04/29/gFcJtf.png#shadow)

现在的任务是输入$w\_{t-n+1},...,w\_{t-1}$这前n-1个单词，然后预测出下一个单词$w\_t$

数学符号说明：

+   $C(i)$：单词$w$对应的词向量，其中$i$为词$w$在整个词汇表中的索引
+   $C$：词向量，大小为$|V|\\times m$的矩阵
+   $|V|$：词汇表的大小，即预料库中去重后的单词个数
+   $m$：词向量的维度，一般大于50
+   $H$：隐藏层的weight
+   $d$：隐藏层的bias
+   $U$：输出层的weight
+   $b$：输出层的bias
+   $W$：输入层到输出层的weight
+   $h$：隐藏层神经元个数

计算流程：

1.  首先将输入的$n-1$个单词索引转为词向量，然后将这$n-1$个向量进行concat，形成一个$(n-1)\\times w$的矩阵，用$X$表示
2.  将$X$送入隐藏层进行计算，$\\text{hidden}\_{out} = \\tanh(d+X\*H)$
3.  输出层共有$|V|$个节点，每个节点$y\_i$表示预测下一个单词$i$的概率，$y$的计算公式为$y=b+X\*W+\\text{hidden}\_{out}\*U$

### 代码实现（PyTorch）

```python
# code by Tae Hwan Jung @graykode, modify by wmathor
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data

dtype = torch.FloatTensor
```

```python
sentences = [ "i like dog", "i love coffee", "i hate milk"]

word_list = " ".join(sentences).split() # ['i', 'like', 'dog', 'dog', 'i', 'love', 'coffee', 'i', 'hate', 'milk']
word_list = list(set(word_list)) # ['i', 'like', 'dog', 'love', 'coffee', 'hate', 'milk']
word_dict = {w: i for i, w in enumerate(word_list)} # {'i':0, 'like':1, 'dog':2, 'love':3, 'coffee':4, 'hate':5, 'milk':6}
number_dict = {i: w for i, w in enumerate(word_list)} # {0:'i', 1:'like', 2:'dog', 3:'love', 4:'coffee', 5:'hate', 6:'milk'}
n_class = len(word_dict) # number of Vocabulary, just like |V|, in this task n_class=7

# NNLM(Neural Network Language Model) Parameter
n_step = len(sentences[0].split())-1 # n-1 in paper, look back n_step words and predict next word. In this task n_step=2
n_hidden = 2 # h in paper
m = 2 # m in paper, word embedding dim
```

由于PyTorch中输入数据是以mini-batch小批量进行的，下面的函数首先将原始数据（词）全部转为索引，然后通过`TensorDataset()`和`DataLoader()`编写一个实用的mini-batch迭代器

```python
def make_batch(sentences):
    input_batch = []
    target_batch = []

    for sen in sentences:
        word = sen.split()
        input = [word_dict[n] for n in word[:-1]] # [0, 1], [0, 3], [0, 5]
        target = word_dict[word[-1]] # 2, 4, 6

        input_batch.append(input) # [[0, 1], [0, 3], [0, 5]]
        target_batch.append(target) # [2, 4, 6]

    return input_batch, target_batch

input_batch, target_batch = make_batch(sentences)
input_batch = torch.LongTensor(input_batch)
target_batch = torch.LongTensor(target_batch)

dataset = Data.TensorDataset(input_batch, target_batch)
loader = Data.DataLoader(dataset=dataset, batch_size=16, shuffle=True)
```

```python
class NNLM(nn.Module):
    def __init__(self):
        super(NNLM, self).__init__()
        self.C = nn.Embedding(n_class, m)
        self.H = nn.Parameter(torch.randn(n_step * m, n_hidden).type(dtype))
        self.W = nn.Parameter(torch.randn(n_step * m, n_class).type(dtype))
        self.d = nn.Parameter(torch.randn(n_hidden).type(dtype))
        self.U = nn.Parameter(torch.randn(n_hidden, n_class).type(dtype))
        self.b = nn.Parameter(torch.randn(n_class).type(dtype))

    def forward(self, X):
        '''
        X: [batch_size, n_step]
        '''
        X = self.C(X) # [batch_size, n_step] => [batch_size, n_step, m]
        X = X.view(-1, n_step * m) # [batch_size, n_step * m]
        hidden_out = torch.tanh(self.d + torch.mm(X, self.H)) # [batch_size, n_hidden]
        output = self.b + torch.mm(X, self.W) + torch.mm(hidden_out, self.U) # [batch_size, n_class]
        return output

model = NNLM()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
```

`nn.Parameter()`的作用是将该参数添加进模型中，使其能够通过`model.parameters()`找到、管理、并且更新。更具体的来说就是：

1.  `nn.Parameter()`与`nn.Module`一起使用时会有一些特殊的属性，其会被自动加到 Module 的`parameters()`迭代器中
2.  使用很简单：`torch.nn.Parameter(data, requires_grad=True)`，其中data为tensor

简单解释一下执行`X=self.C(X)`这一步之后`X`发生了什么变化，假设初始`X=[[0, 1], [0, 3]]`

通过`Embedding()`之后，会将每一个词的索引，替换为对应的词向量，例如`love`这个词的索引是`3`，通过查询Word Embedding表得到行索引为3的向量为`[0.2, 0.1]`，于是就会将原来`X`中`3`的值替换为该向量，所有值都替换完之后，`X=[[[0.3, 0.8], [0.2, 0.4]], [[0.3, 0.8], [0.2, 0.1]]]`

```python
# Training
for epoch in range(5000):
    for batch_x, batch_y in loader:
        optimizer.zero_grad()
        output = model(batch_x)

        # output : [batch_size, n_class], batch_y : [batch_size] (LongTensor, not one-hot)
        loss = criterion(output, batch_y)
        if (epoch + 1)%1000 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

        loss.backward()
        optimizer.step()

# Predict
predict = model(input_batch).data.max(1, keepdim=True)[1]

# Test
print([sen.split()[:n_step] for sen in sentences], '->', [number_dict[n.item()] for n in predict.squeeze()])
```

完整代码：

```python
# code by Tae Hwan Jung @graykode, modify by wmathor
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data

dtype = torch.FloatTensor

sentences = [ "i like dog", "i love coffee", "i hate milk"]

word_list = " ".join(sentences).split() # ['i', 'like', 'dog', 'dog', 'i', 'love', 'coffee', 'i', 'hate', 'milk']
word_list = list(set(word_list)) # ['i', 'like', 'dog', 'love', 'coffee', 'hate', 'milk']
word_dict = {w: i for i, w in enumerate(word_list)} # {'i':0, 'like':1, 'dog':2, 'love':3, 'coffee':4, 'hate':5, 'milk':6}
number_dict = {i: w for i, w in enumerate(word_list)} # {0:'i', 1:'like', 2:'dog', 3:'love', 4:'coffee', 5:'hate', 6:'milk'}
n_class = len(word_dict) # number of Vocabulary, just like |V|, in this task n_class=7

# NNLM(Neural Network Language Model) Parameter
n_step = len(sentences[0].split())-1 # n-1 in paper, look back n_step words and predict next word. In this task n_step=2
n_hidden = 2 # h in paper
m = 2 # m in paper, word embedding dim

def make_batch(sentences):
  input_batch = []
  target_batch = []

  for sen in sentences:
    word = sen.split()
    input = [word_dict[n] for n in word[:-1]] # [0, 1], [0, 3], [0, 5]
    target = word_dict[word[-1]] # 2, 4, 6

    input_batch.append(input) # [[0, 1], [0, 3], [0, 5]]
    target_batch.append(target) # [2, 4, 6]

  return input_batch, target_batch

input_batch, target_batch = make_batch(sentences)
input_batch = torch.LongTensor(input_batch)
target_batch = torch.LongTensor(target_batch)

dataset = Data.TensorDataset(input_batch, target_batch)
loader = Data.DataLoader(dataset=dataset, batch_size=16, shuffle=True)

class NNLM(nn.Module):
  def __init__(self):
    super(NNLM, self).__init__()
    self.C = nn.Embedding(n_class, m)
    self.H = nn.Parameter(torch.randn(n_step * m, n_hidden).type(dtype))
    self.W = nn.Parameter(torch.randn(n_step * m, n_class).type(dtype))
    self.d = nn.Parameter(torch.randn(n_hidden).type(dtype))
    self.U = nn.Parameter(torch.randn(n_hidden, n_class).type(dtype))
    self.b = nn.Parameter(torch.randn(n_class).type(dtype))

  def forward(self, X):
    '''
    X: [batch_size, n_step]
    '''
    X = self.C(X) # [batch_size, n_step] => [batch_size, n_step, m]
    X = X.view(-1, n_step * m) # [batch_size, n_step * m]
    hidden_out = torch.tanh(self.d + torch.mm(X, self.H)) # [batch_size, n_hidden]
    output = self.b + torch.mm(X, self.W) + torch.mm(hidden_out, self.U) # [batch_size, n_class]
    return output

model = NNLM()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training
for epoch in range(5000):
  for batch_x, batch_y in loader:
    optimizer.zero_grad()
    output = model(batch_x)

    # output : [batch_size, n_class], batch_y : [batch_size] (LongTensor, not one-hot)
    loss = criterion(output, batch_y)
    if (epoch + 1)%1000 == 0:
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

    loss.backward()
    optimizer.step()

# Predict
predict = model(input_batch).data.max(1, keepdim=True)[1]

# Test
print([sen.split()[:n_step] for sen in sentences], '->', [number_dict[n.item()] for n in predict.squeeze()])
```

这个代码一开始是在GitHub的一个项目中给出的，下面参考文献给出了链接，代码本身写的没有问题，但是其中有一行注释有问题，就是`X=X.view(-1, n_step*m)`后面的注释，我很确信我写的是正确的。下面两篇参考文献都是一样的错误，需要注意一下

### 参考文献

[A Neural Probabilitic Language Model 论文阅读及实战](https://www.jianshu.com/p/be242ed3f314)

[NLP-tutorial](https://github.com/graykode/nlp-tutorial)


