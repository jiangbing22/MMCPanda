#模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('JYeonn/Vicuna-7b-delta-v0',local_dir='./vicuna_7b-delta_v0')