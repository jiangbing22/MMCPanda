import json
import os
import torch
from code.datasets.sft_dataset import SupervisedDataset
class MCPSupercisedDataset(SupervisedDataset):
    def __init__(self, data_path: str, image_root_path: str,audio_root_path: str):
        with open(data_path,'r') as f:
            json_data = json.load(f)
        
        self.data_list=[]
        for item in json_data:
            sample ={
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
        print(f'[MCPanda] collect {len(self.data_list)} samples for training')
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, i):
        # 返回字典，供 collate 使用
        return self.data_list[i]

    def collate(self, instances):
        # 重写 collate fn
        image_paths = [ins["image_path"] for ins in instances]
        audio_paths = [ins["audio_path"] for ins in instances]
        output_texts = [ins["output_text"] for ins in instances]
        labels = torch.tensor([ins["label"] for ins in instances], dtype=torch.float32)
        
        return dict(
            image_paths=image_paths,
            audio_paths=audio_paths,
            output_texts=output_texts,
            labels=labels
        )
