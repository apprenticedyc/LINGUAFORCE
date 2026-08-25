import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class VirtueDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        
        # 遍历 Virtue DataFrame
        for _, row in dataframe.iterrows():
            # 这里的列名改为 'scenario'
            raw_text = str(row.get("scenario", "")).strip()
            # 这里的列名改为 'label'
            try:
                label = int(row.get("label", 0))
            except:
                continue
            
            # 过滤无效空行
            if not raw_text or raw_text.lower() == 'nan':
                continue
            
            self.samples.append({
                "text": raw_text,
                "label": label
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 处理 [SEP] 分隔符
        # Virtue 数据集通常包含：[情境] [SEP] [特征词]
        # Tokenizer 会自动处理字符串中的 [SEP] 标记，或者你可以手动 split 处理
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
            # 保持与原代码一致的 key 名，确保模型侧代码不用改
            'multi_label': torch.tensor(-100, dtype=torch.long), 
            'binary_label': torch.tensor(sample["label"], dtype=torch.long)
        }

def get_Ethics_Virtue_loaders(args, tokenizer):
    """
    生成 Virtue 数据集的 DataLoader
    """
    print("[*] Loading Virtue datasets...")
    
    # 读取新的文件名
    # 注意：确保 args.train_data_path 等指向的是 virtue_train.csv 等文件
    df_train = pd.read_csv(
        args.train_data_path, 
        sep=',',               # 明确指定逗号分隔
        quotechar='"',         # 明确指定双引号为包裹符
        usecols=[0, 1],        # 【关键】强制只取第 0 列和第 1 列，多出来的全部丢弃
        names=['label', 'scenario'], # 手动指定列名，防止表头解析出错
        header=0,              # 告诉 Pandas 第一行是表头，不要当成数据
        on_bad_lines='skip'    # 如果还是解析不了，直接跳过那一行，别报错
    )
    df_valid = pd.read_csv(
        args.valid_data_path, 
        sep=',',               # 明确指定逗号分隔
        quotechar='"',         # 明确指定双引号为包裹符
        usecols=[0, 1],        # 【关键】强制只取第 0 列和第 1 列，多出来的全部丢弃
        names=['label', 'scenario'], # 手动指定列名，防止表头解析出错
        header=0,              # 告诉 Pandas 第一行是表头，不要当成数据
        on_bad_lines='skip'    # 如果还是解析不了，直接跳过那一行，别报错
    )
    df_test = pd.read_csv(
        args.test_data_path, 
        sep=',',               # 明确指定逗号分隔
        quotechar='"',         # 明确指定双引号为包裹符
        usecols=[0, 1],        # 【关键】强制只取第 0 列和第 1 列，多出来的全部丢弃
        names=['label', 'scenario'], # 手动指定列名，防止表头解析出错
        header=0,              # 告诉 Pandas 第一行是表头，不要当成数据
        on_bad_lines='skip'    # 如果还是解析不了，直接跳过那一行，别报错
    )
    
    print(f"    Train: {len(df_train)} | Valid: {len(df_valid)} | Test: {len(df_test)}")

    # 实例化新的 Dataset
    train_dataset = VirtueDataset(df_train, tokenizer)
    valid_dataset = VirtueDataset(df_valid, tokenizer)
    test_dataset = VirtueDataset(df_test, tokenizer)

    # 封装 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    return train_loader, valid_loader, test_loader