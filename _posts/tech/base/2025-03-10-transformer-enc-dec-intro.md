---
layout: post
title: "Transformer 中 Encoder-Decoder 的区别"
subtitle: "From Attention Is All You Need"
author: "YongQiang"
header-style: text
hidden: false
published: true
catalog: true
tags:
  - Transformer
  - Encoder
  - Decoder
---

### 训练时的区别

什么?

### 推理上的区别

在Transformer模型的推理过程中，编码器（encoder）和解码器（decoder）的执行次数如下：

1. **Encoder（编码器）**：  
   执行**一次**。编码器将整个输入序列（例如源语言句子）一次性处理，生成所有位置的上下文表示。无论输入序列多长，编码器仅需运行一次。

2. **Decoder（解码器）**：  
   执行**N次**，其中**N是输出序列的长度**。解码器以自回归（autoregressive）方式工作，每次生成一个目标token。例如，若输出序列包含10个词，解码器需要运行10次：  
   - 第1次：基于起始符预测第一个词；  
   - 第2次：基于起始符和第一个词预测第二个词；  
   - 依此类推，直到生成结束符或达到最大长度。

**关键细节**：  
- 解码器每次运行时，会利用缓存机制（如保存之前生成的token的Key/Value向量），避免重复计算，提升效率。但缓存不影响执行次数，仅优化计算量。  
- 若使用束搜索（beam search），每条候选路径的解码器执行次数仍由各自生成的序列长度决定，但整体复杂度会增加。

**总结**：  
- **Encoder执行次数 = 1**；  
- **Decoder执行次数 = 输出序列的长度**。


在 Hugging Face 的 Transformers 库中，**Decoder 的执行次数是动态变化的**，由生成的输出序列长度决定，**不需要手动配置**。其执行逻辑与 Transformer 模型的通用推理机制一致（即自回归生成）。以下是具体解释：

---

### 1. **解码器执行次数的动态性**
- **输出长度决定执行次数**：Decoder 的执行次数等于生成的输出序列长度（例如生成 `N` 个 token，则执行 `N` 次）。
- **动态终止条件**：生成可能在达到预设的最大长度前提前终止（例如遇到 `[EOS]` 结束符），因此执行次数可能小于最大配置值。
- **典型场景**：
  ```python
  from transformers import AutoModelForCausalLM, AutoTokenizer

  model = AutoModelForCausalLM.from_pretrained("gpt2")
  tokenizer = AutoTokenizer.from_pretrained("gpt2")

  input_text = "Hello, I am"
  inputs = tokenizer(input_text, return_tensors="pt")

  # 生成输出（假设生成结果为 "Hello, I am a student."，共 5 个 token）
  outputs = model.generate(**inputs, max_new_tokens=20)
  print(tokenizer.decode(outputs[0]))
  ```
  - Decoder 会执行 `5` 次（若生成 5 个 token），每次生成一个 token。

---

### 2. **控制生成过程的参数**
虽然解码器执行次数无法直接配置，但可以通过以下参数间接控制输出长度（从而影响执行次数）：

| 参数                | 作用                                                                 |
|---------------------|----------------------------------------------------------------------|
| `max_new_tokens`    | 限制生成的最大 token 数（直接决定解码器最大执行次数）                |
| `max_length`        | 限制输入+输出的总长度                                                |
| `early_stopping`    | 遇到 `[EOS]` 时提前终止生成（可能减少执行次数）                      |
| `num_beams`         | 束搜索的宽度（增加候选路径，但每条路径的解码次数仍由自身长度决定）   |

---

### 3. **Hugging Face 的内部实现**
- **自回归循环**：在 `model.generate()` 方法内部，通过循环调用 `model.forward()` 逐步生成 token，直到满足终止条件。  
- **缓存优化**：解码器会缓存已生成 token 的 Key/Value 向量（`past_key_values`），避免重复计算，但每次生成仍需执行一次前向传播。

#### 关键代码逻辑（简化）：
```python
# 伪代码：Hugging Face generate() 的简化逻辑
input_ids = initial_input  # 初始输入（如 "[BOS]"）
past_key_values = None

for _ in range(max_new_tokens):
    outputs = model(input_ids, past_key_values=past_key_values)
    logits = outputs.logits
    past_key_values = outputs.past_key_values  # 缓存历史状态
    next_token = select_next_token(logits)     # 根据策略选择下一个 token
    input_ids = append_token(next_token)        # 将新 token 加入输入序列

    if next_token == EOS_TOKEN:
        break
```

---

### 4. **用户常见误解澄清**
- ❌ **误解**：存在类似 `num_decoder_calls` 的参数直接配置解码次数。  
- ✅ **真相**：解码次数完全由输出序列长度动态决定，无法直接设置。唯一可控的是生成终止条件（如 `max_new_tokens`）。

---

### 总结
- **Decoder 执行次数 = 最终生成的 token 数量**（动态变化）。  
- **配置方式**：通过 `max_new_tokens`、`early_stopping` 等参数间接控制生成长度。  
- **优化建议**：若需固定执行次数，可强制设置 `max_new_tokens`，但需注意模型可能生成无意义内容。