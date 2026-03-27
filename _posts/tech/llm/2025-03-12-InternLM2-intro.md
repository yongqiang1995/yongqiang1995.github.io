---
layout: post
title: 大模型 <code>InternLM2</code> 详解
author: "YongQiang"
header-style: text
hidden: false
published: true
tags:
  - InternLM2
  - LLM
  - 模型部署
---



对应模型: https://huggingface.co/internlm/internlm2_5-1_8b-chat

InternLM2 模型结构: LLaMA + GQA

InternLM2模型结构

InternLM2采用了LLaMA的模型结构，并且和LLaMA一样，使用了GQA。LLaMA大家都比较熟悉了，就不再重复说明了。不过比较有意思的是，为了提高Tensor Parallel的效率，InternLM2对Wqkv的权重进行了交织重排.

Ref:
- https://github.com/InternLM/InternLM/blob/main/tools/convert2llama.py
- https://blog.csdn.net/lqfarmer/article/details/140604758
