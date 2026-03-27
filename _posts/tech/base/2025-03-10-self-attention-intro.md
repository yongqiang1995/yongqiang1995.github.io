---
layout: post
title: "理解 <code>Self-Attention</code>"
subtitle: "十分钟掌握 self-attention 的设计原理"
author: "YongQiang"
header-style: text
hidden: false
published: true
catalog: true
# usemathjax: true
mathjax: true
tags:
  - SelfAttention
---

> 受到 信息检索系统 (如搜索引擎) 的启发, 但是它们的数学实现和具体行为有本质区别.

核心思想类比:

在搜索引擎中：用户输入一个 Query ，系统通过匹配 Key （如文档索引），返回对应的 Value （如网页内容）。
在 Transformer 中：每个词（或 token）生成自己的 Query ，通过与其他词的 Key 计算相关性，最终从 Value 中聚合信息。 

设计动机:
Transformer 作者希望模型能像搜索引擎一样， 动态关注输入序列中不同位置的重要信息 ，因此借用了这一术语。

信息检索

✅ Query 是用户输入的问题或需求 (进行分词和清洗, 提取关键词并进行标准化处理, 得到精简 key)
✅ Key 用于匹配 Query，Value 是返回的具体内容
✅ 一个 Key 可以关联多个 Value（多对一）

attention

$$
Attention(Q, K, V) = Softmax(\frac{QK^{T}}{\sqrt{d_{k}}})V
$$

$Q = W_q·x$

$K = W_k·x$

$V = W_v·x$

$x$ 是模型输入.



