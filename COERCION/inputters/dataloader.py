import json
import torch
from torch.utils.data import Dataset, DataLoader

class MoralCoercionDataset(Dataset):
    def __init__(self, data_list, tokenizer, max_length=512):
        """
        初始化数据集
        :param data_list: 包含对话数据的列表
        :param tokenizer: 从外部传入的 tokenizer 实例
        :param max_length: 文本截断的最大长度
        """
        self.data = data_list
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 1. 提取多轮对话并用分隔符拼接
        utterances = item.get("utterances", [])
        # RoBERTa 的分隔符是 </s>，这能帮助模型区分不同轮次的对话
        joined_text = self.tokenizer.sep_token.join(utterances)

        # 2. Tokenize 文本
        encoding = self.tokenizer(
            joined_text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        # 3. 提取标签
        multi_label = item.get("dialog_multi_label", -1)
        binary_label = item.get("dialog_binary_label", -1)

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'multi_label': torch.tensor(multi_label, dtype=torch.long),
            'binary_label': torch.tensor(binary_label, dtype=torch.long)
        }

def load_data(file_path):
    """
    自适应读取 JSON 或 JSONL 格式的数据集
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            # 首先尝试按照标准的 JSON 数组格式读取
            return json.load(f)
        except json.JSONDecodeError:
            # 如果报错，说明可能是 JSON Lines 格式（每行一个 JSON 对象）
            f.seek(0)
            return [json.loads(line) for line in f]

def get_MoralCoercion_loaders(args, tokenizer):
    """
    供 train.py 调用的主函数：一次性生成训练集、验证集和测试集的 DataLoader
    """
    print(f"[*] Loading datasets...")
    
    # 1. 读取数据文件
    train_data = load_data(args.train_data_path)
    valid_data = load_data(args.valid_data_path)
    test_data = load_data(args.test_data_path)
    
    print(f"    Train size: {len(train_data)} | Valid size: {len(valid_data)} | Test size: {len(test_data)}")

    # 2. 实例化 Dataset
    train_dataset = MoralCoercionDataset(train_data, tokenizer)
    valid_dataset = MoralCoercionDataset(valid_data, tokenizer)
    test_dataset = MoralCoercionDataset(test_data, tokenizer)

    # 3. 封装为 DataLoader
    # 训练集打乱顺序 (shuffle=True)，验证集和测试集保持原序 (shuffle=False)
    # pin_memory=True 可以加速数据从 CPU 转移到 GPU
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        pin_memory=True
    )
    
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        pin_memory=True
    )

    return train_loader, valid_loader, test_loader
    # return 0, 0, test_loader