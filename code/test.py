import pickle
import numpy as np
import torch  # 仅用于后续转换示例

# 复用之前的加载函数
def load_mosi_pkl(file_path):
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
    except UnicodeDecodeError:
        with open(file_path, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
    return data

# 加载raw数据并查看类型
mosi_raw = load_mosi_pkl("./data/mosi/mosi.pkl")
print(mosi_raw.keys())
print(mosi_raw["train"][0])  # 取第一条样本
sample=mosi_raw["train"]
# # 打印各核心字段的类型
# print(f"整个raw数据集的类型：{type(mosi_raw)}")  # 通常是 list
# print(f"单条样本的类型：{type(sample)}")        # 通常是 dict
# print(f"文本字段类型：{type(sample['text'])}")  # str（字符串）
# print(f"音频数据类型：{type(sample['audio'])}")    # numpy.ndarray（NumPy数组）
# print(f"标签字段类型：{type(sample['labels'])}")     # float 或 numpy.float64
# print(f"音频数据的shape：{sample['audio'].shape}")    # 1维（原始波形）
# print(f"视频数据的shape：{sample['vision'].shape}") 
# print(f"文本字段的shape：{sample['text'].shape}") 
# print(f"labels字段的shape：{sample['labels'].shape}") 
# print(f"视频数据类型：{type(sample['vision'])}")
