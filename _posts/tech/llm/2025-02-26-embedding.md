---
layout: post
title: 大模型 <code>Embedding</code> 详解
author: "YongQiang"
header-style: text
hidden: false
published: true
tags:
  - Embedding
  - LLM
  - 模型部署
---

![Embedding可视化](https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEgzOCWLMCxaOTTpZKlTr-o3t4l2Uq_IekgKg8P-1nM1OdwonDOiNs0ZCXaMbv12MMTMrstCb2NuKVwS2Oalw7R615Agjr2dkU6-P31BTQxZCHrBAQWJynwROgxFTuHCpwQzTPQauoDd5JQ/s1600/embedding-mnist.gif)

> 本文深入解析 `Embedding` 技术原理, 涵盖文本/图像/视频/音频四大模态的实现方案, 并提供HuggingFace实战代码示例.

## Embedding 简介

所谓 `Embedding` 是将「文本」、「视频图像」、「音频」或其他格式的数据转换为向量的过程.


Embedding是将高维数据映射到低维连续向量空间的技术, 可理解为数据的"DNA序列". 其核心价值体现在：

```python
# 特性验证示例
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

text1 = "深度学习模型"
text2 = "神经网络算法"
emb1 = model.encode(text1)
emb2 = model.encode(text2)

print(f"相似度: {cosine_similarity(emb1, emb2):.2f}")  # 输出: 相似度: 0.86
```

| 特性                | 描述                           |
|---------------------|------------------------------|
| 语义保持性           | 相似内容在向量空间距离相近       |
| 维度可调性           | 典型维度128-4096可配置         |
| 跨模态对齐           | 图文向量可共享空间进行检索       |

## Text Embedding
### 技术原理
通过Transformer网络捕获上下文语义, 典型结构：

```
[CLS] 今天 天气 真 不错 [SEP]
 ↓     ↓    ↓    ↓   ↓     ↓
768维 768  768  768 768   768
```

### 模型推荐

| 模型名称              | 维度 | 语言支持 | 特点                     |
|----------------------|-----|---------|-------------------------|
| BERT-base            | 768 | 中文    | 字粒度编码               |
| text-embedding-3-small| 1536| 多语言  | OpenAI最新优化版         |
| e5-mistral-7b-instruct| 4096| 英文    | 指令微调优化             |

```python
from transformers import AutoModel
model = AutoModel.from_pretrained('BAAI/bge-small-zh-v1.5')
inputs = tokenizer("自然语言处理", return_tensors="pt")
embeddings = model(**inputs).last_hidden_state.mean(dim=1)
```

## Image Embedding
### 编码架构对比
![图像编码架构](https://example.com/image-encoder.png)

```python
import torchvision.models as models
from PIL import Image

resnet = models.resnet50(pretrained=True)
module_list = list(resnet.children())[:-1]
model = torch.nn.Sequential(*module_list)

img = Image.open("cat.jpg").convert('RGB')
emb = model(preprocess(img).unsqueeze(0))
```

| 模型类型       | 特征维度 | 优势领域         |
|--------------|---------|----------------|
| CNN-based    | 2048    | 物体识别        |
| ViT-based    | 768     | 全局语义理解    |
| CLIP         | 512     | 图文跨模态对齐  |

## Video Embedding
### 分层编码方案
```python
# 使用TimeSformer处理视频
from transformers import TimesformerModel

model = TimesformerModel.from_pretrained("facebook/timesformer-base")
video = load_video_frames("demo.mp4")  # shape: (batch, frames, 3, 224, 224)
embeddings = model(video).last_hidden_state
```

处理流程：
1. 帧采样（每秒取3关键帧）
2. 空间特征提取（ResNet-50）
3. 时序建模（LSTM/Transformer）
4. 池化输出（1024维向量）

## Audio Embedding
### 声学特征处理
![音频处理流程](https://example.com/audio-pipeline.png)

```python
import librosa
from transformers import Wav2Vec2Model

y, sr = librosa.load("speech.wav", sr=16000)
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
inputs = processor(y, return_tensors="pt", sampling_rate=16000)
emb = model(**inputs).last_hidden_state.mean(dim=1)
```

| 特征类型      | 维度 | 适用场景         |
|-------------|-----|----------------|
| MFCC        | 40  | 语音识别        |
| Spectrogram | 128 | 声纹识别        |
| Wav2Vec     | 768 | 情感分析        |
```


## Reference

- [CSDN一文搞懂Embedding工作原理](https://blog.csdn.net/qq_39172059/article/details/136661824)