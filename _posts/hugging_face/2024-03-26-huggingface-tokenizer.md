---
layout: post
title: HuggingFace 之 <code>Tokenizer</code>
author: "YongQiang"
header-style: text
hidden: false
published: true
tags:
  - HuggingFace
  - LLM
---

<img src="https://huggingface.co/front/assets/huggingface_logo.svg" 
    alt="HuggingFace Logo" 
    style="width: 120px; margin: 20px auto; display: block;">

## HuggingFace 简介

> `HuggingFace` 是一个领先的人工智能和自然语言处理`(NLP)`平台, 它的 `transformers` 库成为了大规模预训练语言模型的事实标准.<br>
`transformers` 提供了强大的 `API`, 支持 BERT、GPT、T5 等流行模型, 并且可以无缝对接 `PyTorch` 和 `TensorFlow`.<br>
在 `transformers` 库中, `Tokenizer` 负责将文本数据转换为模型可识别的输入格式(例如 token ID、attention mask), 是 `NLP` 任务中不可或缺的部分.

本文将详细介绍 `Tokenizer` 及其使用方法.

## Tokenizer 介绍

`Tokenizer` 是 **NLP** 任务中用于文本预处理的核心组件, 它的主要功能包括: 

- **文本切分(Tokenization)**: 将文本拆分为 token(子词、单词或字符级别)
- **编码(Encoding)**: 将 token 映射为模型可识别的 ID
- **解码(Decoding)**: 将 token ID 还原回可读文本
- **处理特殊 token**(如 `[CLS]`、`[SEP]`、`[PAD]` 等)

`HuggingFace` 提供了多种 `Tokenizer`, 如:

- **WordPiece**(BERT 使用)
- **Byte-Pair Encoding(BPE)**(GPT-2 使用)
- **Unigram**(T5、ALBERT 使用)

## Tokenizer 详解

在 `transformers` 库中, `AutoTokenizer` 提供了一种便捷的方式来加载适用于不同模型的 `Tokenizer`. 

```python
from transformers import AutoTokenizer

# 加载 intfloat 的 Tokenizer
tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-large") # intfloat/multilingual-e5-large, maidalun1020/bce-embedding-base_v1

# 对文本进行 Tokenization
text = "HuggingFace 使 NLP 任务变得简单!"
tokens = tokenizer.tokenize(text)
print("Tokens:", tokens)

# 将 Token 转换为 ID
input_ids = tokenizer.convert_tokens_to_ids(tokens)
print("Token IDs:", input_ids)
```

输出结果:

```bash
Tokens: ['▁Hu', 'gging', 'Fac', 'e', '▁', '使', '▁N', 'LP', '▁', '任务', '变得', '简单', '!']
Token IDs: [7674, 36659, 135518, 13, 6, 4290, 541, 37352, 6, 23906, 36818, 37085, 38]
```

> ❗️注意, 有一些 `Tokenizer` 对中文的支持并不友好, 每一个模型都有自己独有的 `Tokenizer`, 一般在模型发布时同时发出.

### 编码与解码

```python
from transformers import AutoTokenizer

# 加载 intfloat 的 Tokenizer
tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-large") # intfloat/multilingual-e5-large, maidalun1020/bce-embedding-base_v1
text = "HuggingFace 使 NLP 任务变得简单!"
# 直接使用 tokenizer 进行编码
tokenized_output = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
print(tokenized_output)

# 反向解码
decoded_text = tokenizer.decode(tokenized_output["input_ids"][0])
print("Decoded text:", decoded_text)
```

输出结果:

```bash
Encoder text: {'input_ids': tensor([[     0,   7674,  36659, 135518,     13,      6,   4290,    541,  37352,
              6,  23906,  36818,  37085,     38,      2]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])}
Decoded text: <s> HuggingFace 使 NLP 任务变得简单!</s>
```

## 如何自定义 Tokenizer

有时, 默认的 `Tokenizer` 不能满足我们的需求, 例如:
- 处理特定领域的专有术语
- 使用特定的切词规则
- 训练自己的 `Tokenizer`

`HuggingFace` 提供了 `tokenizers` 库, 它是 `transformers` 库的底层组件, 支持高效的自定义 `Tokenizer` 训练. 

### 训练自定义 Tokenizer

```python
from tokenizers import Tokenizer, models, pre_tokenizers, trainers

# 创建一个空的 Tokenizer
custom_tokenizer = Tokenizer(models.BPE())

# 设置 PreTokenizer(拆分规则)
custom_tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

# 训练 Tokenizer
trainer = trainers.BpeTrainer(special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"])
custom_tokenizer.train(["data.txt"], trainer)

# 保存 Tokenizer
custom_tokenizer.save("custom_tokenizer.json")
```

### 加载自定义 Tokenizer

```python
from transformers import PreTrainedTokenizerFast

# 加载自定义 tokenizer
tokenizer = PreTrainedTokenizerFast(tokenizer_file="custom_tokenizer.json")
print(tokenizer.tokenize("HuggingFace is great!"))
```

## Reference

- [HuggingFace 官网](https://huggingface.co/)
- [transformers 官方文档](https://huggingface.co/docs/transformers/index)
- [tokenizers 官方文档](https://huggingface.co/docs/tokenizers/index)

