# PandaGPT 代码架构与训练方法总结

本文档总结了 PandaGPT 的代码结构，将代码实现与模型架构、训练方法一一对应，并给出了加入单模态预测任务（如纯文本任务）的具体做法。

## 1. 代码与模型架构对应 (Code to Model Architecture)

PandaGPT 的核心思想是将多模态编码器 (ImageBind) 的特征通过一个线性投影层 (Linear Projection) 映射到大语言模型 (Vicuna/LLaMA) 的嵌入空间。

### 核心文件
*   **模型定义**: `code/model/openllama.py`
*   **Visual Encoder**: `code/model/ImageBind/`

### 架构组件映射

| 模型组件 (Architecture Component) | 对应代码 (Code Reference) | 说明 |
| :--- | :--- | :--- |
| **整体模型类** | `OpenLLAMAPEFTModel` (in `openllama.py`) | 继承自 `nn.Module`，管理整个前向传播流程。 |
| **多模态编码器 (ImageBind)** | `self.visual_encoder` (lines 85-86) | 加载 ImageBind 巨型模型。在 `__init__` 中被冻结 (`requires_grad = False`)。 |
| **线性投影层 (Linear Projection)** | `self.llama_proj` (lines 113-115) | `nn.Linear` 层，将 ImageBind 的 1024 维特征映射到 LLaMA 的 hidden size (如 4096)。这是 Stage 1 训练的主要参数。 |
| **大语言模型 (LLM)** | `self.llama_model` (lines 104-105) | 加载 Vicuna 或 LLaMA 权重。 |
| **LoRA 适配器** | `peft_config` & `get_peft_model` (lines 95-105) | 使用 HuggingFace `peft` 库在 LLM 的 Attention 层 (`q_proj`, `v_proj` 等) 注入 LoRA 参数。这是 Stage 2 训练的主要参数。 |

### 前向传播流程 (`forward` 方法)
1.  **编码图像**: 调用 `encode_image` -> `visual_encoder` 提取特征 -> `llama_proj` 投影特征。
2.  **构建 Prompt**: 调用 `prompt_wrap` (lines 164-196)，将 `<Image>` token 的 embedding 替换为投影后的图像特征，并拼接文本 embedding。
    *   结构: `[BOS] + [Prompt_Before] + [Image_Embeds] + [Prompt_After]`
3.  **计算 Loss**: 传入 `llama_model` 计算 Causal Language Modeling (CLM) Loss。

---

## 2. 代码与训练方法对应 (Code to Training Method)

PandaGPT 的训练分为两个阶段，通过 `train_sft.py` 和 `dataset` 模块实现。

### 核心文件
*   **训练入口**: `code/train_sft.py`
*   **训练代理**: `code/model/agent.py`
*   **数据加载**: `code/datasets/sft_dataset.py`

### 训练流程映射

| 训练环节 (Training Step) | 对应代码 (Code Reference) | 说明 |
| :--- | :--- | :--- |
| **数据加载** | `SupervisedDataset` (in `sft_dataset.py`) | 读取 JSON 数据，包含 `image_name` 和 `conversation`。 |
| **数据处理** | `build_one_instance` (in `openllama.py`) | 将对话格式化为 `Human: ... Assistant: ...`，并设置 Loss Mask (`target_ids` 中 Human 部分为 -100)。 |
| **DeepSpeed 集成** | `DeepSpeedAgent` (in `agent.py`) | 初始化 DeepSpeed 引擎，管理优化器和分布式训练。 |
| **训练循环** | `main` (in `train_sft.py`, lines 82-89) | 遍历 Epoch 和 Batch，调用 `agent.train_model`。 |
| **参数更新** | `agent.train_model` (in `agent.py`) | 执行 `forward` -> `backward` -> `step`。 |

### 训练阶段 (Stage 1 vs Stage 2)
通过 `code/config/openllama_peft.yaml` 或命令行参数 `--stage` 控制：
*   **Stage 1 (Alignment)**: 仅训练 `llama_proj`。LLM 参数冻结。
*   **Stage 2 (Instruction Tuning)**: 加载 Stage 1 的投影层权重 (`delta_ckpt_path`)，训练 `llama_proj` 和 LoRA 参数。

---

## 3. 加入单模态预测任务的做法 (Adding Single Modality Task)

PandaGPT 目前的代码强依赖于“图像+文本”的输入模式。如果想加入单模态预测任务（例如：**纯文本指令微调** 或 **纯图像分类/描述**），需要修改数据加载和模型的前向传播逻辑。

### 场景 A: 加入纯文本任务 (Text-Only Instruction Tuning)
*目标：让模型既能看图说话，也能回答纯文本问题。*

#### 步骤 1: 修改数据加载 (`code/datasets/sft_dataset.py`)
允许数据中没有图片。

```python
class SupervisedDataset(Dataset):
    def __init__(self, data_path: str, image_root_path: str):
        # ... (现有代码)
        for item in json_data:
            one_image_name = item.get("image_name", None) # 允许为空
            one_caption = item["conversation"]
            
            if one_image_name:
                if not one_image_name.endswith('.jpg'):
                    one_image_name += '.jpg'
                one_image_path = image_root_path + '/{}'.format(one_image_name)
            else:
                one_image_path = None # 标记为 None
                
            self.image_path_list.append(one_image_path)
            self.caption_list.append(one_caption)

    def __getitem__(self, i):
        # 处理 None 的情况，这里返回空字符串或特殊标记给 collate
        img_path = self.image_path_list[i] if self.image_path_list[i] is not None else ""
        return dict(image_paths=img_path, output_texts=self.caption_list[i])
```

#### 步骤 2: 修改模型前向传播 (`code/model/openllama.py`)
在 `forward` 和 `prompt_wrap` 中处理没有图片的情况。

**修改 `forward`:**
```python
def forward(self, inputs):
    image_paths = inputs['image_paths']
    # 区分有图和无图的样本
    # 注意：这里为了简化，假设一个 batch 要么全有图，要么全无图。
    # 如果混合 batch，需要更复杂的 padding 和 mask 处理。
    
    if all(p == "" for p in image_paths):
        img_embeds = None
    else:
        img_embeds, _ = self.encode_image(image_paths)

    output_texts = inputs['output_texts']
    # ... (后续代码)
    inputs_embeds, targets, attention_mask = self.prompt_wrap(img_embeds, input_ids, target_ids, attention_mask)
```

**修改 `prompt_wrap`:**
```python
def prompt_wrap(self, img_embeds, input_ids, target_ids, attention_mask):
    # ... (前部分代码)
    
    # 如果是纯文本任务
    if img_embeds is None:
        # 直接拼接 text embedding，不需要 ImageBind 特征
        # 注意：这里需要根据逻辑去掉 PROMPT_START 中的 <Img> 标记
        
        # 获取纯文本 embedding
        p_after_embeds = self.llama_model.model.model.embed_tokens(input_ids).expand(batch_size, -1, -1)
        bos_embeds = self.llama_model.model.model.embed_tokens(bos)
        
        # 拼接: BOS + Text
        inputs_embeds = torch.cat([bos_embeds, p_after_embeds], dim=1)
        
        # Targets 和 Attention Mask 也对应调整 (去掉 Image 部分的占位)
        empty_targets = torch.ones([batch_size, 1], dtype=torch.long).to(self.device).fill_(-100) # Only BOS
        targets = torch.cat([empty_targets, target_ids], dim=1)
        
        atts_prefix = torch.ones([batch_size, 1], dtype=torch.long).to(self.device)
        attention_mask = torch.cat([atts_prefix, attention_mask], dim=1)
        
        return inputs_embeds, targets, attention_mask

    # ... (原有的多模态拼接逻辑)
```

### 场景 B: 加入纯图像任务 (Image-Only / Classification)
PandaGPT 本质是生成模型。
*   **图像分类**: 构造数据为 `Human: <Img> What is in this image? \n Assistant: A dog.`。这不需要修改代码，只需要构造对应的 Instruction Tuning 数据集即可。
*   **图像嵌入预测**: 如果你想只输入图像，输出图像的 Embedding（不生成文本），则需要修改 `forward` 函数，直接返回 `self.encode_image(image_paths)` 的结果。

### 总结
要加入单模态预测（主要是纯文本），关键在于打破代码中“默认存在图像输入”的假设，在 `Dataset` 加载时允许空图片路径，并在 `OpenLLAMAPEFTModel` 中增加分支逻辑，跳过图像编码和特征拼接步骤。
