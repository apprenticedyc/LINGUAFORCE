import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class SuicideDetectionDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        
        # 遍历 DataFrame 提取数据
        for _, row in dataframe.iterrows():
            raw_text = str(row.get("text", "")).strip()
            raw_label = str(row.get("class", "")).strip().lower()
            
            # 过滤掉无效的空行
            if not raw_text or raw_text.lower() == 'nan':
                continue
                
            # 标签映射：
            # 将 'suicide' 映射为 1 (作为极端情绪/有害特征的 Proxy)
            # 将 'non-suicide' 映射为 0 (安全/正常)
            binary_label = 1 if raw_label == 'suicide' else 0
            
            self.samples.append({
                "text": raw_text,
                "binary_label": binary_label
            })

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
            # 依然将多分类标签设为 -100，让模型专注二分类预训练
            'multi_label': torch.tensor(-100, dtype=torch.long),
            'binary_label': torch.tensor(sample["binary_label"], dtype=torch.long)
        }

def get_SuicideDetection_loaders(args, tokenizer):
    """
    生成 Suicide Detection 的 DataLoader
    假设你已经将原始数据集划分为了 train, valid, test 三个 CSV 文件
    """
    print("[*] Loading Suicide Detection datasets...")
    
    # 使用 pandas 读取 CSV 文件，自动处理多行文本和复杂引号
    df_train = pd.read_csv(args.train_data_path, on_bad_lines='skip')
    df_valid = pd.read_csv(args.valid_data_path, on_bad_lines='skip')
    df_test = pd.read_csv(args.test_data_path, on_bad_lines='skip')
    
    print(f"    Train size: {len(df_train)} | Valid size: {len(df_valid)} | Test size: {len(df_test)}")

    # 实例化 Dataset
    train_dataset = SuicideDetectionDataset(df_train, tokenizer)
    valid_dataset = SuicideDetectionDataset(df_valid, tokenizer)
    test_dataset = SuicideDetectionDataset(df_test, tokenizer)

    # 封装 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    return train_loader, valid_loader, test_loader