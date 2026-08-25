import json
import torch
from torch.utils.data import Dataset, DataLoader

class TalkDownDataset(Dataset):
    def __init__(self, data_list, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        
        for item in data_list:
            # 1. 提取完整的对话上下文
            post = str(item.get("post", "")).strip()
            reply = str(item.get("reply", "")).strip()
            
            # 如果内容完全为空则跳过
            if not post and not reply:
                continue
                
            # 使用分隔符拼接原帖与回复，让模型理解这是对话的交互
            # 格式例如: "原帖内容 </s> 回复内容"
            text = f"{post}{self.tokenizer.sep_token}{reply}"
            
            # 2. 提取布尔类型的标签并映射
            # JSON 中的 true/false 在 Python 解析后会变成原生的 True/False
            raw_label = item.get("label")
            
            if raw_label is True or str(raw_label).lower() == 'true':
                binary_label = 1  # 存在居高临下/说教倾向
            elif raw_label is False or str(raw_label).lower() == 'false':
                binary_label = 0  # 正常交流
            else:
                continue
                
            self.samples.append({
                "text": text,
                "binary_label": binary_label
            })
            
        # --- 诊断信息 ---
        if len(self.samples) == 0:
            print("\n[WARNING] TalkDown Dataset is empty after parsing!")
            if len(data_list) > 0:
                print(f"First raw item:\n{data_list[0]}")

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
            # 同样保持多分类标签为 -100，冻结 7 分类干扰
            'multi_label': torch.tensor(-100, dtype=torch.long),
            'binary_label': torch.tensor(sample["binary_label"], dtype=torch.long)
        }

def load_jsonl(file_path):
    """读取 JSONL 文件的辅助函数"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def get_TalkDown_loaders(args, tokenizer):
    """
    生成 TalkDown 的 DataLoader
    """
    print("[*] Loading TalkDown datasets...")
    
    # 解析 JSONL 数据
    train_data = load_jsonl(args.train_data_path)
    valid_data = load_jsonl(args.valid_data_path)
    test_data = load_jsonl(args.test_data_path)
    
    print(f"    Raw file lines -> Train: {len(train_data)} | Valid: {len(valid_data)}")

    train_dataset = TalkDownDataset(train_data, tokenizer)
    valid_dataset = TalkDownDataset(valid_data, tokenizer)
    test_dataset = TalkDownDataset(test_data, tokenizer)
    
    print(f"    Valid samples after filter -> Train: {len(train_dataset)} | Valid: {len(valid_dataset)} | Test: {len(test_dataset)}")

    if len(train_dataset) == 0:
        raise ValueError("Train dataset is empty! Please check the file path and format.")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)
    
    # 返回占位符以对齐解包逻辑
    return train_loader, valid_loader, test_loader