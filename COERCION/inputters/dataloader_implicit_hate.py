import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class ImplicitHateDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        
        # 遍历 DataFrame 提取数据
        for _, row in dataframe.iterrows():
            raw_text = str(row.get("post", "")).strip()
            raw_label = str(row.get("class", "")).strip().lower()
            
            # 过滤无效空行
            if not raw_text or raw_text.lower() == 'nan':
                continue
                
            # 标签映射：not_hate -> 0, 另外两类 hate -> 1
            if raw_label == 'not_hate':
                binary_label = 0
            elif raw_label in ['implicit_hate', 'explicit_hate']:
                binary_label = 1
            else:
                continue # 如果有异常标签则跳过
            
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
            # 同样将多分类标签设为 -100，冻结 7 分类损失
            'multi_label': torch.tensor(-100, dtype=torch.long),
            'binary_label': torch.tensor(sample["binary_label"], dtype=torch.long)
        }

def get_ImplicitHate_loaders(args, tokenizer):
    """
    生成 Implicit Hate 的 DataLoader
    """
    print("[*] Loading Implicit Hate datasets...")
    
    # 依然使用 pandas 读取，防止带引号的句子导致错列
    df_train = pd.read_csv(args.train_data_path, on_bad_lines='skip')
    df_valid = pd.read_csv(args.valid_data_path, on_bad_lines='skip')
    df_test = pd.read_csv(args.test_data_path, on_bad_lines='skip')
    
    print(f"    Train size: {len(df_train)} | Valid size: {len(df_valid)} | Test size: {len(df_test)}")

    # 实例化 Dataset
    train_dataset = ImplicitHateDataset(df_train, tokenizer)
    valid_dataset = ImplicitHateDataset(df_valid, tokenizer)
    test_dataset = ImplicitHateDataset(df_test, tokenizer)

    # 封装 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    return train_loader, valid_loader, test_loader