import torch
from torch.utils.data import Dataset
import json
from transformers import BertTokenizer  # 添加导入
from datasets import load_dataset as hf_load_dataset
import os

# 设置HuggingFace镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# 如果需要设置代理
# os.environ['TRANSFORMERS_OFFLINE'] = '1'  # 离线模式
# os.environ['HF_HOME'] = '/path/to/cache/huggingface'  # 缓存目录

class HFDataset(Dataset):
    """包装 Hugging Face 数据集的自定义数据集类"""
    def __init__(self, dataset_name, block_size=1024, max_samples=5000000):
        # 使用镜像加载数据集
        full_dataset = hf_load_dataset(
            dataset_name,
            trust_remote_code=True,  
            use_auth_token=False,    
        )['train']
        
        # 如果指定了最大样本数，则只使用部分数据
        if max_samples is not None:
            max_samples = min(max_samples, len(full_dataset))
            self.raw_dataset = full_dataset.select(range(max_samples))
            print(f"从原始数据集（{len(full_dataset)}条）中选择了 {max_samples} 条数据")
        else:
            self.raw_dataset = full_dataset
        
        # 初始化BERT分词器
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        self.block_size = block_size
        
        # 添加column_names属性
        self.column_names = self.raw_dataset.column_names
        
    def __len__(self):
        return len(self.raw_dataset)
    
    def __getitem__(self, idx):
        text = self.raw_dataset[idx]['text']
        # 使用BERT分词器进行编码
        encoded = self.tokenizer.encode(
            text,
            add_special_tokens=True,
            max_length=self.block_size + 1,
            truncation=True,
            padding='max_length'
        )
        
        # 创建输入和目标
        x = torch.tensor(encoded[:-1], dtype=torch.long)
        y = torch.tensor(encoded[1:], dtype=torch.long)
        return x, y

def load_dataset(dataset_name):
    """加载数据集的包装函数"""
    try:
        # 加载数据集
        raw_dataset = hf_load_dataset(
            dataset_name,
            trust_remote_code=True,
            use_auth_token=False,
        )
        
        # 打印数据集信息
        print("Dataset structure:", raw_dataset)
        print("Available splits:", raw_dataset.keys())
        if 'train' in raw_dataset:
            print("Train split size:", len(raw_dataset['train']))
            print("Sample:", raw_dataset['train'][0])
        
        # 根据数据集结构选择合适的分割
        if 'train' in raw_dataset:
            return raw_dataset
        else:
            return {'train': raw_dataset}
            
    except Exception as e:
        print(f"加载数据集时出错: {e}")
        raise

class MyDataset(Dataset):
    def __init__(self, path, block_size=512):
        # 使用BERT分词器
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        self.block_size = block_size
        
        # 特殊符号用于分割训练文本
        self.sep_token = self.tokenizer.sep_token_id

        self.encoded_data = []
        self.max_lines = 5000000
        raw_data = []

        print(f"开始加载数据...")
        with open(path, 'r') as f:
            for i, line in enumerate(f):
                if i >= self.max_lines:
                    break
                if i % 10000 == 0:
                    print(f"已处理 {i} 行数据...")
                try:
                    text = json.loads(line.strip())['text']
                    raw_data.append(text)
                except json.JSONDecodeError as je:
                    print(f"JSON解析错误，跳过行 {i}: {je}")
                    continue
                except Exception as e:
                    print(f"未知错误，跳过行 {i}: {e}")
                    continue

        # 预处理文本
        full_encoded = []
        for text in raw_data:
            encoded_text = self.tokenizer.encode(text, add_special_tokens=True)
            full_encoded.extend(encoded_text + [self.sep_token])

        # 把长文本切割为block_size的文本块
        for i in range(0, len(full_encoded), self.block_size):
            chunk = full_encoded[i:i + self.block_size + 1]
            if len(chunk) < self.block_size + 1:
                chunk = chunk + [self.tokenizer.pad_token_id] * (self.block_size + 1 - len(chunk))
            self.encoded_data.append(chunk)
        
        print(f"数据预处理完成，共处理了 {len(self.encoded_data)} 个文本块")

    def encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=True)

    def decode(self, ids):
        return self.tokenizer.decode(ids)

def validate_sequence_length(text_length, block_size):
    """验证序列长度是否合适"""
    if text_length > block_size:
        print(f"警告：输入序列长度 ({text_length}) 超过了 block_size ({block_size})")
    return min(text_length, block_size) 