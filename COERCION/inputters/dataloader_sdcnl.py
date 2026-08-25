import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class SDCNLDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        
        # 获取列名以进行安全校验
        columns = dataframe.columns.tolist()
        
        for _, row in dataframe.iterrows():
            # 1. 提取文本：优先使用 megatext_clean，如果没有则拼接 title 和 selftext
            text = str(row.get("megatext_clean", "")).strip()
            if not text or text.lower() == 'nan':
                title = str(row.get("title", "")).strip()
                selftext = str(row.get("selftext", "")).strip()
                # 如果都没有内容，则跳过
                if (not title or title.lower() == 'nan') and (not selftext or selftext.lower() == 'nan'):
                    continue
                text = f"{title}. {selftext}"
                
            # 2. 提取并映射标签 (处理多种可能的格式)
            raw_label = str(row.get("is_suicide", "")).strip().lower()
            
            # 映射为 1 (存在自杀倾向/极端负面情绪)
            if raw_label in ['1', '1.0', 'true', 'suicide']:
                binary_label = 1
            # 映射为 0 (安全/普通抑郁/无自杀倾向)
            elif raw_label in ['0', '0.0', 'false', 'non-suicide', 'non_suicide', 'depression']:
                binary_label = 0
            else:
                continue # 遇到无法解析的标签则跳过
            
            self.samples.append({
                "text": text,
                "binary_label": binary_label
            })
            
        # --- 诊断信息 ---
        if len(self.samples) == 0:
            print(f"\n[WARNING] SDCNL Dataset is empty after parsing!")
            print(f"Found columns: {columns}")
            print(f"First row raw data:\n{dataframe.head(1).to_dict('records')}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Tokenize
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
            # 保持 7分类任务屏蔽，专注二分类
            'multi_label': torch.tensor(-100, dtype=torch.long),
            'binary_label': torch.tensor(sample["binary_label"], dtype=torch.long)
        }

def get_SDCNL_loaders(args, tokenizer):
    """
    生成 SDCNL 的 DataLoader
    """
    print("[*] Loading SDCNL datasets...")
    
    # 读取 CSV，自动跳过格式损坏的行
    df_train = pd.read_csv(args.train_data_path, on_bad_lines='skip', engine='python')
    df_valid = pd.read_csv(args.valid_data_path, on_bad_lines='skip', engine='python')
    df_test = pd.read_csv(args.test_data_path, on_bad_lines='skip', engine='python')
    
    print(f"    Raw file lines -> Train: {len(df_train)} | Valid: {len(df_valid)}")

    train_dataset = SDCNLDataset(df_train, tokenizer)
    valid_dataset = SDCNLDataset(df_valid, tokenizer)
    test_dataset = SDCNLDataset(df_test, tokenizer)
    
    print(f"    Valid samples after filter -> Train: {len(train_dataset)} | Valid: {len(valid_dataset)} | Test: {len(test_dataset)}")

    if len(train_dataset) == 0:
        raise ValueError("Train dataset is empty! Please check the printed diagnostic warnings.")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)
    
    # 同样返回占位符以对齐解包逻辑
    return train_loader, valid_loader, test_loader