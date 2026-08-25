import json
import torch
from torch.utils.data import Dataset, DataLoader

class ToxiChatDataset(Dataset):
    def __init__(self, data_list, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        
        # 解析 ToxiChat 数据
        for item in data_list:
            # 1. 提取多轮对话的历史上下文 (Thread)
            thread = item.get("reddit_thread", [])
            context_utterances = [turn["text"].strip() for turn in thread]
            
            # 2. 分别提取 dgpt 和 gpt3 的回复，构建为两个独立的样本
            if "final_dgpt_response" in item:
                self._add_sample(context_utterances, item["final_dgpt_response"])
                
            if "final_gpt3_response" in item:
                self._add_sample(context_utterances, item["final_gpt3_response"])

    def _add_sample(self, context_utterances, response_dict):
        """将上下文与机器人的单次回复拼接，并提取二分类标签"""
        # 拼接对话历史与当前回复
        all_utterances = context_utterances + [response_dict["text"].strip()]
        # 用 </s> 连接
        joined_text = self.tokenizer.sep_token.join(all_utterances)
        
        # 标签映射：ToxiChat 中 "Safe" 视为 0 (非绑架/非冒犯)，其他 (如 Offensive) 视为 1
        offense_label = response_dict.get("offense_label", "Safe")
        binary_label = 0 if offense_label.lower() == "safe" else 1
        
        self.samples.append({
            "text": joined_text,
            "binary_label": binary_label
        })

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
            # 【关键】将多分类标签设为 -100，PyTorch 损失函数会自动忽略它
            'multi_label': torch.tensor(-100, dtype=torch.long),
            'binary_label': torch.tensor(sample["binary_label"], dtype=torch.long)
        }

def load_jsonl(file_path):
    """读取 ToxiChat 的 jsonl 文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def get_ToxiChat_loaders(args, tokenizer):
    """
    生成 ToxiChat 的 DataLoader
    """
    print("[*] Loading ToxiChat datasets...")
    
    train_data = load_jsonl(args.train_data_path)
    # 假设你有验证集和测试集，如果没有，可以用 train_test_split 切分
    valid_data = load_jsonl(args.valid_data_path) 
    test_data = load_jsonl(args.test_data_path)
    
    train_dataset = ToxiChatDataset(train_data, tokenizer)
    valid_dataset = ToxiChatDataset(valid_data, tokenizer)
    test_dataset = ToxiChatDataset(test_data, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    return train_loader, valid_loader, test_loader