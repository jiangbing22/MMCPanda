import json
import os
import pandas as pd
import torch
from code.datasets.sft_dataset import SupervisedDataset

class MCPSupervisedDataset(SupervisedDataset):
    def __init__(self, data_path: str, image_root_path: str, audio_root_path: str, mode: str = 'train'):
        """
        Args:
            data_path: 路径指向 .json 或 .csv (mosi/label.csv)
            image_root_path: 图片根目录
            audio_root_path: 音频根目录
            mode: 'train', 'dev', 'test'，用于过滤 CSV 数据
        """
        self.data_list = []
        
        if data_path.endswith('.csv'):
            # === 读取 MOSI CSV 格式 ===
            # CSV 格式假设: id, text, label, mode
            print(f"[MCPanda] Loading MOSI data from CSV: {data_path} (mode={mode})")
            df = pd.read_csv(data_path)
            
            # 过滤 split (mode)
            # CSV 中的 mode 列通常是 'train', 'valid'/'dev', 'test'
            # 简单的模糊匹配处理
            if mode == 'dev':
                df = df[df['mode'].isin(['valid', 'dev', 'validation'])]
            else:
                df = df[df['mode'] == mode]
                
            for index, row in df.iterrows():
                video_id = str(row['id']) # e.g. "03bSnISJMiM_1"
                text = row['text']
                label = float(row['label'])
                
                # 构造路径，MOSI 通常以 ID 命名文件
                # 假设 visual 数据在 raw/video/ID.jpg 或类似位置，需根据实际情况调整后缀
                # 如果没有图片，image_path 设为 None，模型会自动填充 dummy
                img_path = os.path.join(image_root_path, f"{video_id}.jpg")
                if not os.path.exists(img_path):
                    # 尝试其他后缀或置空
                    img_path = None 
                
                # 构造音频路径
                aud_path = os.path.join(audio_root_path, f"{video_id}.wav")
                if not os.path.exists(aud_path):
                    # 警告或置空
                    # print(f"Warning: Audio not found for {video_id}")
                    aud_path = None

                sample = {
                    "image_path": img_path,
                    "audio_path": aud_path,
                    "output_text": text,
                    "label": label,
                    "id": video_id
                }
                self.data_list.append(sample)
                
        else:
            # === 原有的 JSON 读取逻辑 ===
            print(f"[MCPanda] Loading data from JSON: {data_path}")
            with open(data_path, 'r') as f:
                json_data = json.load(f)
            
            for item in json_data:
                sample = {
                    "image_path": None,
                    "audio_path": None,
                    "output_text": item["conversation"], 
                    "label": float(item.get("label", 0.0))
                }

                if "image_name" in item and item['image_name']:
                    img_name = item['image_name']
                    if not img_name.endswith('.jpg'):
                        img_name += '.jpg'
                    sample['image_path'] = os.path.join(image_root_path, img_name)
                    
                if "audio_name" in item and item['audio_name']:
                    audio_name = item['audio_name']
                    if not audio_name.endswith('.wav'):
                        audio_name += '.wav'
                    sample['audio_path'] = os.path.join(audio_root_path, audio_name)

                self.data_list.append(sample)
                
        print(f'[MCPanda] Collected {len(self.data_list)} samples.')

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, i):
        return self.data_list[i]

    def collate(self, instances):
        # 批量处理逻辑
        image_paths = [ins["image_path"] for ins in instances]
        audio_paths = [ins["audio_path"] for ins in instances]
        output_texts = [ins["output_text"] for ins in instances]
        
        # 将 output_texts 封装为符合模型输入的格式 (如 list of dict)
        # 注意：这里的 output_texts 只是纯文本，OpenLLAMA 模型通常需要 [{"from": "human", "value": ...}, ...] 格式
        # 如果原始 text 只是句子，我们需要包装一下
        formatted_conversations = []
        for text in output_texts:
            # 简单的包装，模拟对话格式，以便 process_batch_instance 处理
            if isinstance(text, str):
                formatted_conversations.append([
                    {"from": "human", "value": "Analyze the sentiment of this video/audio."},
                    {"from": "gpt", "value": text} # 这里如果是生成任务，Text通常是 Target
                ])
            else:
                formatted_conversations.append(text)

        labels = torch.tensor([ins["label"] for ins in instances], dtype=torch.float32)
        
        return dict(
            image_paths=image_paths,
            audio_paths=audio_paths,
            output_texts=formatted_conversations,
            labels=labels
        )