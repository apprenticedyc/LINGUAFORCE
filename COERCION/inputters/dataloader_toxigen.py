import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class ToxiGenDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        
        # 遍历 DataFrame 提取数据
        for _, row in dataframe.iterrows():
            raw_text = str(row.get("text", ""))
            raw_label = str(row.get("label", "")).strip().lower()
            
            # 1. 清洗数据：去除 b'...' 或 b"..." 的残留格式
            cleaned_text = self._clean_byte_string(raw_text)
            
            # 过滤掉清洗后为空的无效数据
            if not cleaned_text:
                continue
                
            # 2. 标签映射
            # ToxiGen 的标签主要是 'hate' 和 'neutral'
            binary_label = 1 if raw_label == 'hate' else 0
            
            self.samples.append({
                "text": cleaned_text,
                "binary_label": binary_label
            })

    def _clean_byte_string(self, text):
        """去除 CSV 中残留的 b'...' 或 b"..." 格式"""
        text = text.strip()
        if text.startswith("b'") and text.endswith("'"):
            return text[2:-1]
        elif text.startswith('b"') and text.endswith('"'):
            return text[2:-1]
        return text

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Tokenize 文本
        encoding = self.tokenizer(
            sample["text"],
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            # 继续保持多分类标签为 -100，让模型在这个阶段专注于二分类任务
            'multi_label': torch.tensor(-100, dtype=torch.long),
            'binary_label': torch.tensor(sample["binary_label"], dtype=torch.long)
        }

def get_ToxiGen_loaders(args, tokenizer):
    """
    生成 ToxiGen 的 DataLoader
    """
    print("[*] Loading ToxiGen datasets...")
    
    # 使用 pandas 读取 CSV 文件
    # 遇到解析错误 (如数据中未闭合的引号) 时跳过错误行
    df_train = pd.read_csv(args.train_data_path, on_bad_lines='skip')
    df_valid = pd.read_csv(args.valid_data_path, on_bad_lines='skip')
    df_test = pd.read_csv(args.test_data_path, on_bad_lines='skip')
    
    print(f"    Train size: {len(df_train)} | Valid size: {len(df_valid)} | Test size: {len(df_test)}")

    # 实例化 Dataset
    train_dataset = ToxiGenDataset(df_train, tokenizer)
    valid_dataset = ToxiGenDataset(df_valid, tokenizer)
    test_dataset = ToxiGenDataset(df_test, tokenizer)

    # 封装 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    return train_loader, valid_loader, test_loader