---
layout: post
title: "chatglm tuning 实战"
subtitle: ''
author: "YiKe"
header-style: text
tags:
- impl
---

今天跑了一下这个[chatglm tuning](https://github.com/mymusise/ChatGLM-Tuning)
记录下遇到的问题

# 网络问题
## ssl
  要么是公司信任机制问题，要么是huggingface的证书问题，反正会verify失败。只能跳过认证。
```python
  import os
  os.environ['CURL_CA_BUNDLE'] = ''
```
  这里直接把CA置为空，把验证过程给阻断了

## timeout
   模型稳健非常大，如果一直hang,就会timeout，目前试下来best practice 是不断重试
```python
while True:
    try:
        model = AutoModel.from_pretrained("THUDM/chatglm-6b", cache_dir = cache_dir,resume_download=True,load_in_8bit=True, trust_remote_code=True, device_map='auto')

        tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b",cache_dir = cache_dir, trust_remote_code=True,resume_download=True,)
    except Exception as e:
        print("异常")
        print(e)
```
    这里需要设置resume_download和cache_dir

## gcc 版本升级

参考[stackoverflow](https://stackoverflow.com/questions/36327805/how-to-install-gcc-5-3-with-yum-on-centos-7-2)
