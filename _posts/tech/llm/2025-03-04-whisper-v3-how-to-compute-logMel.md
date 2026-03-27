---
layout: post
title: How to compute the log-Mel spectrogram?
author: "YongQiang"
header-style: text
hidden: false
published: true
tags:
  - Whisper
  - Mel
  - 模型部署
---

```
past_key_value是在Transformer中的self-attention模块用于处理序列数据时，记录之前时间步的键（key）和值（value）状态。在处理较长的序列或者将模型应用于生成任务（如文本生成）时，它可以提高计算效率。

在生成任务中，模型会逐个生成新的单词。每生成一个新单词，模型就需要处理包含新单词的序列。通过使用 past_key_value，我们可以避免在每个时间步重新计算整个序列的键和值，而只需在前一时间步的基础上计算新单词的键和值。这样，我们可以节省计算资源，从而加速生成过程。

如果 past_key_value 不是 None，则将新的键和值状态与之前的键和值状态拼接在一起。这样，我们就可以利用以前的计算结果，在新的时间步上仅计算新单词的键和值。最后，更新后的键和值状态被存储在 past_key_value 中，以备在下一个时间步使用。
```


```python
# audio.mp3 -> log_mel 计算过程
# lib/python3.9/site-packages/transformers/models/whisper/processing_whisper.py#69
inputs = self.feature_extractor(audio, *args, sampling_rate=sampling_rate, **kwargs)

# 下面是 feature_extractor 的具体实现过程
# 其中 audio 是 librosa.load 在 sampling_rate=16k 的结果, 目前 shape 为 (2862144,)
# self.feature_extractor -> lib/python3.9/site-packages/transformers/models/whisper/feature_extraction_whisper.py(248)
raw_speech = [np.asarray([raw_speech]).T] # raw_speech[0].shape (2862144, 1)
batched_speech = BatchFeature({"input_features": raw_speech})
padded_inputs = self.pad(batched_speech, padding=padding, ...) # 282
# padded_inputs.input_features[0].shape (480000, 1)
input_features = padded_inputs.get("input_features").transpose(2, 0, 1) # (1, 1, 480000)
input_features = extract_fbank_features(input_features[0], device) # extract_fbank_features 可以是 torch 实现, 也可以是 numpy 实现, shape (1, 128, 3000)
padded_inputs["input_features"] = input_features
# return_tensors 'pt'
padded_inputs = padded_inputs.convert_to_tensors(return_tensors) # 转换后: [0.8952, 0.6014, 0.4266]; 转换前: [0.8951664, 0.6014253, 0.4266312]. 显示精度会有区别 torch.float32
# 最后转为 torch.float16 会有精度丢失
# fp32 [0.8952, 0.6014, 0.4266] -> fp16 [0.8950, 0.6016, 0.4265]
```

模型生成转译结果

```python
# input_features.shape: torch.Size([1, 128, 3000])
# encoder_outputs['last_hidden_state'].shape: torch.Size([1, 1500, 1280])
# decoder_input_ids.shape: torch.Size([1, N])
# outputs['last_hidden_state'].shape: torch.Size([1, N, 1280])
lm_logits = self.proj_out(outputs['last_hidden_state']) # torch.Size([1, N, 51866]) # 这里的 N 似乎会变
# 在 model.generate 函数中, 会不断计算输出结果的 ids? 很奇怪

# 下面具体的代码执行流程
# lib/python3.9/site-packages/transformers/models/whisper/generation_whisper.py(478)generate()
# outputs = super().generate(input_features, ...) 587, input_features.shape torch.Size([1, 128, 3000])
# 上面这行调用: /home/baiyongqiang/miniforge-pypy3/envs/hf/lib/python3.9/site-packages/transformers/generation/utils.py#1539 generate 函数
# TODO: model_kwargs["attention_mask"].shape torch.Size([1, 128]), 这里的 attention_mask 是什么?
# encoder 被隐式调用: lib/python3.9/site-packages/transformers/generation/utils.py(502)_prepare_encoder_decoder_kwargs_for_generation()
# 定义 GenerationMixin 的对象, 通过 encoder = self.get_encoder() 调用 encoder 推理
# model_kwargs 中包含了 encoder outputs 等一些重要信息
model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(
    inputs_tensor, model_kwargs, model_input_name, generation_config
)
input_ids, model_kwargs = self._prepare_decoder_input_ids_for_generation(model_kwargs=model_kwargs, ...)
# input_ids: tensor([[50258, 50260, 50360]], device='cuda:0')
# decoder_input_ids: tensor([[50258, 50260, 50360]], device='cuda:0') 不清楚初始这三个值表示的是什么含义
# decoder_start_token_id: tensor(50258, device='cuda:0')
# lib/python3.9/site-packages/transformers/generation/utils.py(2566)_sample() 这个函数貌似做 decoder, 这里面有一个 while 循环
# while 中会调用一个 self 函数, 这个函数实际上走的是 lib/python3.9/site-packages/transformers/models/whisper/modeling_whisper.py(1746)forward(), 这里面会有一个 outputs = self.model(...)#1754 函数
# outputs = self.model(...)#1754 会调用 /home/baiyongqiang/miniforge-pypy3/envs/hf/lib/python3.9/site-packages/transformers/models/whisper/modeling_whisper.py(1602)forward() 函数
# modeling_whisper.py(1602)forward() 函数中调用 decoder: decoder_outputs = self.decoder(), decoder_outputs.keys() dict_keys(['last_hidden_state', 'past_key_values'])
# decoder_outputs['last_hidden_state'].shape torch.Size([1, 3, 1280]) TODO: 第一次的输出. 而第一次的输入为 input_ids=decoder_input_ids: tensor([[50258, 50260, 50360]], device='cuda:0')
# len(decoder_outputs['past_key_values']) 为 4, shape 也比较复杂
# lm_logits = self.proj_out(outputs[0]) # 对 decoder 的输出做一个线性变换 torch.Size([1, 3, 1280]) -> torch.Size([1, 3, 51866]), 其中 51866 是 vocabulary size
# past_key_values 会被保存
next_token_logits = outputs.logits[:, -1, :].clone() # 表示模型在生成序列的最后一个位置对词表中每个 token 的预测分数 (未归一化的logits)
next_token_scores = logits_processor(input_ids, next_token_logits) # 看起来是预测了下一个 token, 词汇表中最大的概率值可能就是下一个值 TODO: 我可能需要手动解析一下 tokenizer
<!>
  (Pdb) input_ids
  tensor([[50258, 50260, 50360]], device='cuda:0')
  (Pdb) outputs.logits.shape
  torch.Size([1, 3, 51866])
  (Pdb) next_token_logits
  tensor([[ -7.5430,  -7.2266, -10.1719,  ...,  -9.8047,  -7.5312, -10.3516]],
        device='cuda:0', dtype=torch.float16)
  (Pdb) input_ids
  tensor([[50258, 50260, 50360]], device='cuda:0')
  (Pdb) next_token_scores
  tensor([[-inf, -inf, -inf,  ..., -inf, -inf, -inf]], device='cuda:0',
        dtype=torch.float16)
  (Pdb) next_token_scores.max()
  tensor(4.8203, device='cuda:0', dtype=torch.float16)
<!>
next_tokens = torch.argmax(next_token_scores, dim=-1) # 找到 next_tokens 的索引位置 (即 ids)
(Pdb) next_tokens
tensor([50365], device='cuda:0')
# 如果还没有到达 EOS 停止规则
next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences) # 这里 pad_token_id: tensor(50257, device='cuda:0'), 不懂在干嘛
input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1) # input_ids 会和下一个预测的 token 做拼接
(Pdb) input_ids
tensor([[50258, 50260, 50360, 50365]], device='cuda:0')

# 继续 while 循环 (第二次)
decoder_input_ids: tensor([[50365]], device='cuda:0') # 很神奇, 之前那三个值不见了, 但是 input_ids 一直保存最初的那三个值
(Pdb) outputs.logits.shape
torch.Size([1, 1, 51866])
(Pdb) input_ids
tensor([[50258, 50260, 50360, 50365, 11914]], device='cuda:0')
# 继续 while 循环 (第三次)
(Pdb) model_inputs['decoder_input_ids'] # 似乎每一次都只有一个值?
tensor([[11914]], device='cuda:0')
(Pdb) input_ids
tensor([[50258, 50260, 50360, 50365, 11914, 30246]], device='cuda:0')
# 继续 while 循环 (第 N 次), TODO: 这里的 N 是由谁来控制的?
##################################3
# 高级封装
generated_ids = model.generate(
    input_features,
    return_timestamps=True,          # 启用时间戳
    forced_decoder_ids=forced_decoder_ids,  # 强制指定语言
    # max_new_tokens=448,              # 与原始配置保持一致
    max_length=448,
    compression_ratio_threshold=1.35,
    no_speech_threshold=0.6
) # torch.Size([1, 105])
# 后处理: 解码输出 (包含时间戳处理), 其实是利用 tokenizer 解析模型输出的 ids
transcription = processor.batch_decode(
    generated_ids, 
    return_timestamps=True,
    output_offsets=True
)[0]

```


> TODO: 实现音频到 梅尔图谱的计算, 2025.03.05 20.33

## Whisper 30s 限制
Whisper 模型在训练时就将音频预先切分成大约 30 秒的片段，其原因主要有以下几点：

1. **训练数据标准化**  
   Whisper 使用了 680,000 小时的有监督数据，这些数据在预处理阶段被切分成固定长度的 30 秒音频片段。这样可以使模型在训练时始终看到相同长度的输入，方便学习从梅尔频谱到文本的映射，同时降低了模型输入长度不一带来的复杂性。citeturn0search11

2. **Transformer 架构的计算限制**  
   Whisper 的核心是基于 Transformer 的编码器-解码器架构。Transformer 的自注意力机制在计算时复杂度为 O(N²)，其中 N 是时间步数。对于 30 秒的音频（经过转换后大约有 3000 个时间帧），已经是一个比较合理的序列长度；而更长的输入会显著增加内存消耗和计算负担，可能导致推理速度下降或内存不足。citeturn0search11

3. **时间戳和连续性问题**  
   长时间音频的连续输入可能导致时间戳预测不准确，甚至出现“幻觉”现象（例如重复、漂移等问题）。通过将音频切块处理，可以在每个片段内更好地控制和调整时间戳，最后再将各个片段的转录结果拼接起来，确保整体转录的准确性和连续性。citeturn0academia10

综上所述，30 秒的限制既符合训练数据的预处理方式，也兼顾了模型计算效率和转录效果，是 Whisper 设计上的一项折衷与优化。


## 梅尔频谱

梅尔频谱是一种将音频信号转换成频率图像的方法，它利用了人耳对不同频率的敏感程度。其主要步骤如下：

1. **短时傅里叶变换（STFT）**  
   将连续的音频信号分成小块（通常用25毫秒的窗口），对每一块进行傅里叶变换，得到局部的频谱信息。

2. **梅尔滤波器组**  
   通过一组设计好的滤波器（这些滤波器在低频部分更密集，高频部分更稀疏），将传统的线性频谱映射到“梅尔刻度”。这种刻度更加符合人耳对音高的感知，因为人耳对低频变化更敏感，而对高频变化不那么敏感。

3. **取对数**  
   对滤波器组输出的能量取对数，以获得对数梅尔频谱，这样可以更好地处理音频信号的动态范围。

最终得到的梅尔频谱图就是一个二维数组，其中一维表示时间，另一维表示梅尔频段，数值则代表各个频段在每个时间窗口内的对数能量。它在语音识别、音乐信息检索等任务中非常常用，因为它更好地反映了人类听觉系统的特点。citeturn0search11

