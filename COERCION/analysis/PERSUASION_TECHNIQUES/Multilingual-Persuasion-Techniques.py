# from transformers import pipeline, AutoTokenizer

# model_path = "D:/model/multilingual_persuasion_techniques" 

# try:
#     # 1. 先单独加载 Tokenizer，并强制禁用 fast 模式
#     tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    
#     # 2. 将加载好的 tokenizer 传给 pipeline
#     classifier = pipeline("text-classification", model=model_path, tokenizer=tokenizer)
    
#     test_text = "You must buy this product now before it's too late!"
#     results = classifier(test_text)
#     print("预测结果:", results)

# except Exception as e:
#     print(f"加载或运行模型时出错: {e}")


import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 1. 基础配置
model_path = "D:/model/multilingual_persuasion_techniques"          # 您的模型文件夹路径
input_file = "inputters/data/data.jsonl"                            # 您的输入文件名
output_file = "inputters/data/data_persuasion_techniques.jsonl"     # 输出的预测结果文件名
threshold = 0.5                                                     # 多标签预测的概率阈值 (大于0.5判定为存在该技巧)

# 2. 加载模型与分词器
print("正在加载模型与分词器...")
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"模型已加载，当前使用设备: {device}")

# 3. 处理文件并进行预测
print("开始处理整段对话数据...")
with open(input_file, 'r', encoding='utf-8') as fin, \
     open(output_file, 'w', encoding='utf-8') as fout:
    
    for line in fin:
        if not line.strip(): continue
            
        data = json.loads(line)
        dialogue_id = data.get("dialogue_id")
        utterances = data.get("utterances", [])
        
        # --- 核心修改：拼接整段对话 ---
        full_dialogue_text = ""
        for i, utt in enumerate(utterances):
            speaker = "Person1" if i % 2 == 0 else "Person2"
            # 拼接格式例如: "Person1: xxx. Person2: yyy."
            full_dialogue_text += f"{speaker}: {utt} "
        
        full_dialogue_text = full_dialogue_text.strip()
        
        # 如果对话为空，直接跳过
        if not full_dialogue_text:
            continue
            
        # 对整段文本进行编码 (开启截断，防止超过512 token报错)
        inputs = tokenizer(full_dialogue_text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # 模型推理
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            
        # 多标签处理：使用 Sigmoid 将输出映射到 0-1 之间
        probs = torch.sigmoid(logits[0])
        
        # 找出所有概率大于阈值 (0.5) 的标签
        predicted_indices = (probs > threshold).nonzero(as_tuple=True)[0].tolist()
        predicted_labels = [model.config.id2label[idx] for idx in predicted_indices]
        
        # 兜底逻辑：如果全段对话都没检测出明显技巧，取最高概率的那个或标为 None
        if not predicted_labels:
            best_idx = torch.argmax(probs).item()
            if probs[best_idx] < 0.3: 
                predicted_labels = ["None"]
            else:
                predicted_labels = [model.config.id2label[best_idx]]

        # (在原本预测代码的组装结果部分，加上 dialog_binary_label)
        dialog_binary_label = data.get("dialog_binary_label") # 读取原标签
        
        output_data = {
            "dialogue_id": dialogue_id,
            "dialog_binary_label": dialog_binary_label, # 把标签传给输出文件
            "full_text": full_dialogue_text,
            "predicted_techniques": predicted_labels
        }
        fout.write(json.dumps(output_data, ensure_ascii=False) + '\n')

print(f"预测完成！结果已保存至 {output_file}")