# PandaMCP 实现细节与修改文档

本文档详细记录了在 **PandaGPT** 基础上集成 **KAN-MCP** 和 **MMPareto** 思想以适应 **MOSI** 情感分析任务的修改工作。

## 1. 核心思路

我们的目标是改进 PandaGPT，使其能够更好地处理多模态情感分析任务（如 MOSI）。主要改进点如下：

1.  **引入单模态辅助任务 (Unimodal Auxiliary Tasks)**：
    *   在原有的语言模型生成任务（LM Loss）基础上，增加了针对 Audio、Visual 和 Text 三个模态的单模态情感预测任务。
    *   利用 **MIB (Multi-Information Bottleneck)** 作为单模态任务的 Head，旨在提取更鲁棒的模态特定特征。
2.  **MMPareto 梯度协调 (Gradient Adjustment)**：
    *   引入 MMPareto 的思想来解决多任务学习（LM 任务 + 单模态任务）中的梯度冲突问题。
    *   特别关注共享参数（LLaMA 的 LoRA 层）的梯度平衡。

## 2. 代码修改详解

### 2.1 新增模型文件 `code/model/panda_mcp.py`

创建了新的模型类 `PandaMCP` 和代理类 `PandaMCPAgent`。

#### `PandaMCP` 类
*   **继承关系**：继承自 `OpenLLAMAPEFTModel`，复用其 LLaMA + LoRA 的基础架构。
*   **新增组件**：
    *   `self.audio_proj` / `self.visual_proj`: 将 MOSI 数据集的原始特征（Audio: 74维, Visual: 47维）投影到 LLaMA 的 Hidden Size。
    *   `self.TEncoder` / `self.AEncoder` / `self.VEncoder`: 三个 `mib` 模块，分别用于 Text, Audio, Visual 的单模态情感回归预测。**注意**：`AEncoder` 和 `VEncoder` 的输入维度已修改为 `text_dim` (LLaMA Hidden Size)，以接收投影后的特征。
*   **前向传播逻辑 (`forward`)**：
    1.  **特征编码**：
        *   对 Visual 和 Audio 特征进行 Mean Pooling 并投影得到 `v_embeds`, `a_embeds`。
        *   拼接成 Multimodal Embeddings。
    2.  **LLaMA 生成**：
        *   将 Multimodal Embeddings 作为 Prefix 拼接到 Text Embeddings 前。
        *   计算 `loss_lm` (Next Token Prediction)。
    3.  **单模态预测**：
        *   提取 LLaMA 最后一层 Hidden States 中对应 Text 的特征。
        *   **关键修改**：分别将 Text 特征, `a_embeds`, `v_embeds` 送入对应的 `mib` Encoder。使用投影后的 `embeds` 作为输入是为了确保单模态任务的梯度能够回传并更新 `audio_proj` 和 `visual_proj` 层。
        *   计算 `loss_t`, `loss_a`, `loss_v`。
    4.  **输出**：返回包含 4 个 Loss 的字典。

#### `PandaMCPAgent` 类
*   **继承关系**：继承自 `DeepSpeedAgent`。
*   **训练逻辑 (`train_model`)**：
    *   执行前向传播获取 4 个 Loss。
    *   **MMPareto 集成 (完整实现)**：
        *   **共享参数识别**：自动识别模型中的 **LoRA 参数** 以及 **Audio/Visual Projector 参数** 作为多任务共享参数集合。
        *   **独立梯度计算**：通过循环对每个 Loss (`lm`, `t`, `a`, `v`) 执行 `backward(retain_graph=True)`，并提取共享参数的梯度向量。
        *   **权重求解**：调用 `MinNormSolver.find_min_norm_element` 计算最优的 Loss 加权系数，目标是找到各任务梯度的最小范数凸组合，从而缓解梯度冲突。
        *   **加权更新**：使用计算出的权重对 Loss 进行加权求和，执行最终的反向传播和参数更新。
        *   若未启用 MMPareto (`--use_mmpareto` 为 False) 或无共享参数，则回退到等权重求和。

### 2.2 数据集适配 `code/datasets/mosi_dataset.py`

*   **新增 `MOSIDataset` 类**：
    *   专门用于加载 `mosi.pkl` 数据。
    *   实现了 `__getitem__` 以提取 text, visual, acoustic, label。
    *   实现了 `collate_fn` 对时序特征进行 Padding，以适配 Batch 训练。

### 2.3 训练脚本 `code/train_mosi.py`

*   基于 `train_sft.py` 修改而来。
*   移除了 ImageBind 和 Vicuna 的复杂路径依赖，简化为适配 MOSI 任务的配置。
*   默认使用 `panda_mcp` 模型配置。
*   配置了 DeepSpeed 环境。

### 2.4 配置文件修改

*   **`code/config/base.yaml`**: 注册了 `panda_mcp` 模型及其对应的 Dataset 类。
*   **`code/dsconfig/panda_mcp_stage_1.json`**: 复制并重命名了 DeepSpeed 配置文件。
*   **`code/model/__init__.py`**: 导入并注册了新模型。

## 3. 为什么做这些修改？(Design Rationale)

1.  **为什么保留 LoRA？**
    *   MOSI 数据集相对较小（约 1k-2k 样本），全量微调 7B/13B 的 LLaMA 容易过拟合且显存开销巨大。LoRA 是最经济高效的选择。
    *   MMPareto 的作用域主要就在这些共享的 LoRA 参数上。

2.  **为什么使用 MIB？**
    *   MIB (Multi-Information Bottleneck) 能够有效地过滤模态中的噪声，提取与任务（情感分析）最相关的信息。这在 KAN-MCP 中已经被证明是有效的。

3.  **关于 MMPareto 的实现**
    *   为了实现完整的 MMPareto 算法，我们移除了 DeepSpeed 依赖，改用标准 PyTorch 训练循环。
    *   这使得我们能够自由控制 `backward` 过程，准确计算每个任务对共享参数（LoRA + Projectors）的独立梯度，并应用 MinNormSolver 计算出的最优权重。
    *   这确保了模型在优化 LLaMA 主干和模态投影层时，能够平衡 LM 任务和单模态辅助任务之间的竞争。

## 4. 后续优化方向

*   **特征聚合**：目前的 Mean Pooling 比较简单，可以尝试 Attention Pooling 或 LSTM/Transformer Layer 来处理 Audio/Visual 时序特征。
*   **MMPareto 完整版**：尝试在非 DeepSpeed 环境下（或深入 DeepSpeed 源码）实现真正的梯度重加权。
*   **KAN 集成**：目前的 MIB 内部还是 MLP，可以尝试将 MIB 中的 MLP 替换为 KAN (Kolmogorov-Arnold Networks) 层，进一步提升表达能力（参考 KAN-MCP）。


