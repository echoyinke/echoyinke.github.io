---
layout: post
title: "BERT 的 PyTorch 实现"
subtitle: 'BERT'
author: "YiKe"
header-style: text
tags:
- impl
---



本文主要介绍一下如何使用 PyTorch 复现BERT。

### 准备数据集

这里我并没有用什么大型的数据集，而是手动输入了两个人的对话，主要是为了降低代码阅读难度，我希望读者能更关注模型实现的部分

```python
'''
  code by Tae Hwan Jung(Jeff Jung) @graykode, modify by wmathor
  Reference : https://github.com/jadore801120/attention-is-all-you-need-pytorch
         https://github.com/JayParks/transformer, https://github.com/dhlee347/pytorchic-bert
'''
import re
import math
import torch
import numpy as np
from random import *
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data

text = (
    'Hello, how are you? I am Romeo.\n' # R
    'Hello, Romeo My name is Juliet. Nice to meet you.\n' # J
    'Nice meet you too. How are you today?\n' # R
    'Great. My baseball team won the competition.\n' # J
    'Oh Congratulations, Juliet\n' # R
    'Thank you Romeo\n' # J
    'Where are you going today?\n' # R
    'I am going shopping. What about you?\n' # J
    'I am going to visit my grandmother. she is not very well' # R
)
sentences = re.sub("[.,!?\\-]", '', text.lower()).split('\n') # filter '.', ',', '?', '!'
word_list = list(set(" ".join(sentences).split())) # ['hello', 'how', 'are', 'you',...]
word2idx = {'[PAD]' : 0, '[CLS]' : 1, '[SEP]' : 2, '[MASK]' : 3}
for i, w in enumerate(word_list):
    word2idx[w] = i + 4
idx2word = {i: w for i, w in enumerate(word2idx)}
vocab_size = len(word2idx)

token_list = list()
for sentence in sentences:
    arr = [word2idx[s] for s in sentence.split()]
    token_list.append(arr)
```

最终token\_list是个二维的list，里面每一行代表一句话

```python
print(token_list)
'''
[[12, 7, 22, 5, 39, 21, 15],
 [12, 15, 13, 35, 10, 27, 34, 14, 19, 5],
 [34, 19, 5, 17, 7, 22, 5, 8],
 [33, 13, 37, 32, 28, 11, 16],
 [30, 23, 27],
 [6, 5, 15],
 [36, 22, 5, 31, 8],
 [39, 21, 31, 18, 9, 20, 5],
 [39, 21, 31, 14, 29, 13, 4, 25, 10, 26, 38, 24]]
'''
```

### 模型参数

```python
# BERT Parameters
maxlen = 30
batch_size = 6
max_pred = 5 # max tokens of prediction
n_layers = 6
n_heads = 12
d_model = 768
d_ff = 768*4 # 4*d_model, FeedForward dimension
d_k = d_v = 64  # dimension of K(=Q), V
n_segments = 2
```

+   `maxlen`表示同一个batch中的所有句子都由30个token组成，不够的补PAD（这里我实现的方式比较粗暴，直接固定所有batch中的所有句子都为30）
+   `max_pred`表示最多需要预测多少个单词，即BERT中的完形填空任务
+   `n_layers`表示Encoder Layer的数量
+   `d_model`表示Token Embeddings、Segment Embeddings、Position Embeddings的维度
+   `d_ff`表示Encoder Layer中全连接层的维度
+   `n_segments`表示Decoder input由几句话组成

### 数据预处理

数据预处理部分，我们需要根据概率随机make或者替换（以下统称mask）一句话中15%的token，还需要拼接任意两句话

```python
# sample IsNext and NotNext to be same in small batch size
def make_data():
    batch = []
    positive = negative = 0
    while positive != batch_size/2 or negative != batch_size/2:
        tokens_a_index, tokens_b_index = randrange(len(sentences)), randrange(len(sentences)) # sample random index in sentences
        tokens_a, tokens_b = token_list[tokens_a_index], token_list[tokens_b_index]
        input_ids = [word2idx['[CLS]']] + tokens_a + [word2idx['[SEP]']] + tokens_b + [word2idx['[SEP]']]
        segment_ids = [0] * (1 + len(tokens_a) + 1) + [1] * (len(tokens_b) + 1)

        # MASK LM
        n_pred =  min(max_pred, max(1, int(len(input_ids) * 0.15))) # 15 % of tokens in one sentence
        cand_maked_pos = [i for i, token in enumerate(input_ids)
                          if token != word2idx['[CLS]'] and token != word2idx['[SEP]']] # candidate masked position
        shuffle(cand_maked_pos)
        masked_tokens, masked_pos = [], []
        for pos in cand_maked_pos[:n_pred]:
            masked_pos.append(pos)
            masked_tokens.append(input_ids[pos])
            if random() < 0.8:  # 80%
                input_ids[pos] = word2idx['[MASK]'] # make mask
            elif random() > 0.9:  # 10%
                index = randint(0, vocab_size - 1) # random index in vocabulary
                while index < 4: # can't involve 'CLS', 'SEP', 'PAD'
                  index = randint(0, vocab_size - 1)
                input_ids[pos] = index # replace

        # Zero Paddings
        n_pad = maxlen - len(input_ids)
        input_ids.extend([0] * n_pad)
        segment_ids.extend([0] * n_pad)

        # Zero Padding (100% - 15%) tokens
        if max_pred > n_pred:
            n_pad = max_pred - n_pred
            masked_tokens.extend([0] * n_pad)
            masked_pos.extend([0] * n_pad)

        if tokens_a_index + 1 == tokens_b_index and positive < batch_size/2:
            batch.append([input_ids, segment_ids, masked_tokens, masked_pos, True]) # IsNext
            positive += 1
        elif tokens_a_index + 1 != tokens_b_index and negative < batch_size/2:
            batch.append([input_ids, segment_ids, masked_tokens, masked_pos, False]) # NotNext
            negative += 1
    return batch
# Proprecessing Finished

batch = make_data()
input_ids, segment_ids, masked_tokens, masked_pos, isNext = zip(*batch)
input_ids, segment_ids, masked_tokens, masked_pos, isNext = \
    torch.LongTensor(input_ids),  torch.LongTensor(segment_ids), torch.LongTensor(masked_tokens),\
    torch.LongTensor(masked_pos), torch.LongTensor(isNext)

class MyDataSet(Data.Dataset):
  def __init__(self, input_ids, segment_ids, masked_tokens, masked_pos, isNext):
    self.input_ids = input_ids
    self.segment_ids = segment_ids
    self.masked_tokens = masked_tokens
    self.masked_pos = masked_pos
    self.isNext = isNext
  
  def __len__(self):
    return len(self.input_ids)
  
  def __getitem__(self, idx):
    return self.input_ids[idx], self.segment_ids[idx], self.masked_tokens[idx], self.masked_pos[idx], self.isNext[idx]

loader = Data.DataLoader(MyDataSet(input_ids, segment_ids, masked_tokens, masked_pos, isNext), batch_size, True)
```

上述代码中，`positive`变量代表两句话是连续的个数，`negative`代表两句话不是连续的个数，我们需要做到在一个batch中，这两个样本的比例为1:1。随机选取的两句话是否连续，只要通过判断`tokens_a_index + 1 == tokens_b_index`即可

然后是随机mask一些token，`n_pred`变量代表的是即将mask的token数量，`cand_maked_pos`代表的是有哪些位置是候选的、可以mask的（因为像\[SEP\]，\[CLS\]这些不能做mask，没有意义），最后`shuffle()`一下，然后根据`random()`的值选择是替换为`[MASK]`还是替换为其它的token

接下来会做两个Zero Padding，第一个是为了补齐句子的长度，使得一个batch中的句子都是相同长度。第二个是为了补齐mask的数量，因为不同句子长度，会导致不同数量的单词进行mask，我们需要保证同一个batch中，mask的数量（必须）是相同的，所以也需要在后面补一些没有意义的东西，比方说`[0]`

以上就是整个数据预处理的部分

### 模型构建


```python
def get_attn_pad_mask(seq_q, seq_k):
    batch_size, seq_len = seq_q.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_q.data.eq(0).unsqueeze(1)  # [batch_size, 1, seq_len]
    return pad_attn_mask.expand(batch_size, seq_len, seq_len)  # [batch_size, seq_len, seq_len]

def gelu(x):
    """
      Implementation of the gelu activation function.
      For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
      0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
      Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class Embedding(nn.Module):
    def __init__(self):
        super(Embedding, self).__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)  # token embedding
        self.pos_embed = nn.Embedding(maxlen, d_model)  # position embedding
        self.seg_embed = nn.Embedding(n_segments, d_model)  # segment(token type) embedding
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, seg):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long)
        pos = pos.unsqueeze(0).expand_as(x)  # [seq_len] -> [batch_size, seq_len]
        embedding = self.tok_embed(x) + self.pos_embed(pos) + self.seg_embed(seg)
        return self.norm(embedding)

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k) # scores : [batch_size, n_heads, seq_len, seq_len]
        scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is one.
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)
    def forward(self, Q, K, V, attn_mask):
        # q: [batch_size, seq_len, d_model], k: [batch_size, seq_len, d_model], v: [batch_size, seq_len, d_model]
        residual, batch_size = Q, Q.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # q_s: [batch_size, n_heads, seq_len, d_k]
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # k_s: [batch_size, n_heads, seq_len, d_k]
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1,2)  # v_s: [batch_size, n_heads, seq_len, d_v]

        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1) # attn_mask : [batch_size, n_heads, seq_len, seq_len]

        # context: [batch_size, n_heads, seq_len, d_v], attn: [batch_size, n_heads, seq_len, seq_len]
        context = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v) # context: [batch_size, seq_len, n_heads * d_v]
        output = nn.Linear(n_heads * d_v, d_model)(context)
        return nn.LayerNorm(d_model)(output + residual) # output: [batch_size, seq_len, d_model]

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_ff) -> (batch_size, seq_len, d_model)
        return self.fc2(gelu(self.fc1(x)))

class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask) # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs: [batch_size, seq_len, d_model]
        return enc_outputs

class BERT(nn.Module):
    def __init__(self):
        super(BERT, self).__init__()
        self.embedding = Embedding()
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Dropout(0.5),
            nn.Tanh(),
        )
        self.classifier = nn.Linear(d_model, 2)
        self.linear = nn.Linear(d_model, d_model)
        self.activ2 = gelu
        # fc2 is shared with embedding layer
        embed_weight = self.embedding.tok_embed.weight
        self.fc2 = nn.Linear(d_model, vocab_size, bias=False)
        self.fc2.weight = embed_weight

    def forward(self, input_ids, segment_ids, masked_pos):
        output = self.embedding(input_ids, segment_ids) # [bach_size, seq_len, d_model]
        enc_self_attn_mask = get_attn_pad_mask(input_ids, input_ids) # [batch_size, maxlen, maxlen]
        for layer in self.layers:
            # output: [batch_size, max_len, d_model]
            output = layer(output, enc_self_attn_mask)
        # it will be decided by first token(CLS)
        h_pooled = self.fc(output[:, 0]) # [batch_size, d_model]
        logits_clsf = self.classifier(h_pooled) # [batch_size, 2] predict isNext

        masked_pos = masked_pos[:, :, None].expand(-1, -1, d_model) # [batch_size, max_pred, d_model]
        h_masked = torch.gather(output, 1, masked_pos) # masking position [batch_size, max_pred, d_model]
        h_masked = self.activ2(self.linear(h_masked)) # [batch_size, max_pred, d_model]
        logits_lm = self.fc2(h_masked) # [batch_size, max_pred, vocab_size]
        return logits_lm, logits_clsf
model = BERT()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adadelta(model.parameters(), lr=0.001)
```

这段代码中用到了一个激活函数`gelu`，这是BERT论文中提出来的，具体公式可以看这篇文章[GELU激活函数](https://blog.csdn.net/liruihongbob/article/details/86510622)

这段代码有一个特别不好理解的地方，就是到数第7行的代码，用到了`torch.gather()`函数，这里我稍微讲一下。这个函数实际上实现了以下的功能

```python
out = torch.gather(input, dim, index)
# out[i][j][k] = input[index[i][j][k]][j][k] # dim=0
# out[i][j][k] = input[i][index[i][j][k]][k] # dim=1
# out[i][j][k] = input[i][j][index[i][j][k]] # dim=2
```

具体以一个例子来说就是，首先我生成`index`变量

```python
index = torch.from_numpy(np.array([[1, 2, 0], [2, 0, 1]])).type(torch.LongTensor)
index = index[:, :, None].expand(-1, -1, 10)
print(index)
'''
tensor([[[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
         [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],

        [[2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]])
'''
```

然后随机生成一个\[2, 3, 10\]维的tensor，可以理解为有2个batch，每个batch有3句话，每句话由10个词构成，只不过这里的词不是以正整数（索引）的形式出现，而是连续的数值

```python
input = torch.rand(2, 3, 10)
print(input)
'''
tensor([[[0.7912, 0.7098, 0.7548, 0.8627, 0.1966, 0.6327, 0.6629, 0.8158,
          0.7094, 0.1476],
         [0.0774, 0.6794, 0.0030, 0.1855, 0.7391, 0.0641, 0.2950, 0.9734,
          0.7018, 0.3370],
         [0.2190, 0.3976, 0.0112, 0.5581, 0.1329, 0.2154, 0.6277, 0.0850,
          0.4446, 0.5158]],

        [[0.4145, 0.8486, 0.9515, 0.3826, 0.6641, 0.5192, 0.2311, 0.6960,
          0.4215, 0.5597],
         [0.0221, 0.5232, 0.3971, 0.8972, 0.2772, 0.5046, 0.1881, 0.9044,
          0.6925, 0.9837],
         [0.6797, 0.5538, 0.8139, 0.1199, 0.0095, 0.4940, 0.7814, 0.1484,
          0.0200, 0.7489]]])
'''
```

之后调用`torch.gather(input, 1, index)`函数

```python
print(torch.gather(input, 1, index))
'''
tensor([[[0.0774, 0.6794, 0.0030, 0.1855, 0.7391, 0.0641, 0.2950, 0.9734,
          0.7018, 0.3370],
         [0.2190, 0.3976, 0.0112, 0.5581, 0.1329, 0.2154, 0.6277, 0.0850,
          0.4446, 0.5158],
         [0.7912, 0.7098, 0.7548, 0.8627, 0.1966, 0.6327, 0.6629, 0.8158,
          0.7094, 0.1476]],

        [[0.6797, 0.5538, 0.8139, 0.1199, 0.0095, 0.4940, 0.7814, 0.1484,
          0.0200, 0.7489],
         [0.4145, 0.8486, 0.9515, 0.3826, 0.6641, 0.5192, 0.2311, 0.6960,
          0.4215, 0.5597],
         [0.0221, 0.5232, 0.3971, 0.8972, 0.2772, 0.5046, 0.1881, 0.9044,
          0.6925, 0.9837]]])
'''
```

`index`中第一行的tensor会作用于`input`的第一个batch，具体来说，原本三句话的顺序是\[0, 1, 2\]，现在会根据\[1, 2, 0\]调换顺序。`index`中第2行的tensor会作用于`input`的第二个batch，具体来说，原本三句话的顺序是\[0, 1, 2\]，现在会根据`[2, 0, 1]`调换顺序

### 训练&测试

以下是训练代码

```python
for epoch in range(180):
    for input_ids, segment_ids, masked_tokens, masked_pos, isNext in loader:
      logits_lm, logits_clsf = model(input_ids, segment_ids, masked_pos)
      loss_lm = criterion(logits_lm.view(-1, vocab_size), masked_tokens.view(-1)) # for masked LM
      loss_lm = (loss_lm.float()).mean()
      loss_clsf = criterion(logits_clsf, isNext) # for sentence classification
      loss = loss_lm + loss_clsf
      if (epoch + 1) % 10 == 0:
          print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
```

以下是测试代码

```python
# Predict mask tokens ans isNext
input_ids, segment_ids, masked_tokens, masked_pos, isNext = batch[0]
print(text)
print([idx2word[w] for w in input_ids if idx2word[w] != '[PAD]'])

logits_lm, logits_clsf = model(torch.LongTensor([input_ids]), \
                 torch.LongTensor([segment_ids]), torch.LongTensor([masked_pos]))
logits_lm = logits_lm.data.max(2)[1][0].data.numpy()
print('masked tokens list : ',[pos for pos in masked_tokens if pos != 0])
print('predict masked tokens list : ',[pos for pos in logits_lm if pos != 0])

logits_clsf = logits_clsf.data.max(1)[1].data.numpy()[0]
print('isNext : ', True if isNext else False)
print('predict isNext : ',True if logits_clsf else False)
```

Github 项目地址：[nlp-tutorial](https://github.com/wmathor/nlp-tutorial)

