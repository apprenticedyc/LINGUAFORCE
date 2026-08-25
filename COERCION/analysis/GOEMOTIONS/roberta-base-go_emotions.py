import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 1. 基础配置
model_name = "D:/model/roberta-base-go_emotions"
input_file = "inputters/data/data.jsonl"                    # 您的原始输入文件名
output_file = "analysis/GOEMOTIONS/emotion_predictions.jsonl"     # 包含情感预测的结果文件
threshold = 0.3                               # GoEmotions的标签较多，阈值设为0.3比较容易捕捉到微弱情感

print("正在加载 GoEmotions 模型与分词器...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"模型已加载，当前使用设备: {device}")

# 定义一个辅助函数来进行预测
def predict_emotions(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        
    probs = torch.sigmoid(logits[0])
    predicted_indices = (probs > threshold).nonzero(as_tuple=True)[0].tolist()
    predicted_labels = [model.config.id2label[idx] for idx in predicted_indices]
    
    # 兜底：如果没过阈值，取概率最高的一个
    if not predicted_labels:
        best_idx = torch.argmax(probs).item()
        predicted_labels = [model.config.id2label[best_idx]]
        
    return predicted_labels

print("开始进行情感分析...")
with open(input_file, 'r', encoding='utf-8') as fin, \
     open(output_file, 'w', encoding='utf-8') as fout:
    
    for line in fin:
        if not line.strip(): continue
        data = json.loads(line)
        
        dialogue_id = data.get("dialogue_id")
        utterances = data.get("utterances", [])
        dialog_binary_label = data.get("dialog_binary_label")
        
        # --- 1. 单句情感预测 ---
        utterance_emotions = []
        full_dialogue_text = ""
        
        for i, utt in enumerate(utterances):
            speaker = "Person1" if i % 2 == 0 else "Person2"
            full_dialogue_text += f"{speaker}: {utt} "
            
            # 预测单句话的情感
            emotions = predict_emotions(utt)
            utterance_emotions.append({
                "text": utt,
                "emotions": emotions
            })
            
        # --- 2. 整段对话情感预测 ---
        full_dialogue_text = full_dialogue_text.strip()
        dialogue_emotions = predict_emotions(full_dialogue_text) if full_dialogue_text else []
        
        # 组装结果
        output_data = {
            "dialogue_id": dialogue_id,
            "dialog_binary_label": dialog_binary_label,
            "utterance_emotions_analysis": utterance_emotions, # 单句级别的预测结果列表
            "dialogue_emotions": dialogue_emotions             # 整段级别的预测结果
        }
        fout.write(json.dumps(output_data, ensure_ascii=False) + '\n')

print(f"情感预测完成！结果已保存至 {output_file}")