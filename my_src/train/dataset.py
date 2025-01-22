import torch
from torch.utils.data import Dataset
import json
import tiktoken
from datasets import load_dataset as hf_load_dataset
import os

# 设置HuggingFace镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# 如果需要设置代理
# os.environ['TRANSFORMERS_OFFLINE'] = '1'  # 离线模式
# os.environ['HF_HOME'] = '/path/to/cache/huggingface'  # 缓存目录

class HFDataset(Dataset):
    """包装 Hugging Face 数据集的自定义数据集类"""
    def __init__(self, dataset_name, block_size=512):
        # 使用镜像加载数据集
        self.dataset = hf_load_dataset(
            dataset_name,
            trust_remote_code=True,  # 允许运行远程代码
            use_auth_token=False,    # 不使用认证
        )['train']
        
        # 初始化tokenizer
        self.enc = tiktoken.get_encoding("gpt2")
        self.block_size = block_size
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        text = self.dataset[idx]['text']
        encoded = self.enc.encode(text)
        
        # 确保长度为block_size + 1
        if len(encoded) > self.block_size + 1:
            encoded = encoded[:self.block_size + 1]
        elif len(encoded) < self.block_size + 1:
            encoded = encoded + [self.enc.eot_token] * (self.block_size + 1 - len(encoded))
        
        # 创建输入和目标
        x = torch.tensor(encoded[:-1], dtype=torch.long)
        y = torch.tensor(encoded[1:], dtype=torch.long)
        return x, y

def load_dataset(dataset_name):
    """加载数据集的包装函数"""
    return HFDataset(dataset_name)

class MyDataset(Dataset):
    def __init__(self, path, block_size=512):
        self.enc = tiktoken.get_encoding("gpt2")
        self.block_size = block_size
        self.eos_token = self.enc.encode(
            "<|endoftext|>",
            allowed_special={"<|endoftext|>"}
        )[0]

        self.encoded_data = []
        self.max_lines = 1000
        raw_data = []
        
        with open(path, 'r') as f:
            for i, line in enumerate(f):
                if i >= self.max_lines:
                    break
                try:
                    text = json.loads(line.strip())['text']
                    raw_data.append(text)
                except (json.JSONDecodeError, Exception):
                    continue
                    
        full_encoded = []
        for text in raw_data:
            encoded_text = self.enc.encode(text)
            full_encoded.extend(encoded_text + [self.eos_token])
        
        for i in range(0, len(full_encoded), self.block_size):
            chunk = full_encoded[i:i+self.block_size+1]
            if len(chunk) < self.block_size + 1:
                chunk = chunk + [self.eos_token] * (self.block_size + 1 - len(chunk))
            self.encoded_data.append(chunk)
    
    def __len__(self):
        return len(self.encoded_data)
    
    def __getitem__(self, idx):
        chunk = self.encoded_data[idx]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

    def encode(self, text):
        return self.enc.encode(text)

    def decode(self, ids):
        return self.enc.decode(ids) 