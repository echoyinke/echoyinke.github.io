---
layout: post
title: "Attention复杂度"
subtitle: ''
author: "YiKe"
header-style: text
tags:
- impl
---




#### 实现

这里主要是针对苏神所提出的方法进行实现，但是由于笔者本人水平有限，因此最终实现的代码当中其实存在一些问题，主要是：

1.  从测试结果来看，改进后的计算速度并没有提升
2.  无法做到求和为1

代码实现主要是针对BERT的PyTorch实现这篇文章的代码，更具体的说，其实仅修改了`ScaledDotProductAttention`这个函数，因此下面只放出这部分代码

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