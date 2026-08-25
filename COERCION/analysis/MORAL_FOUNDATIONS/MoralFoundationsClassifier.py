import json
import torch
from transformers import RobertaForSequenceClassification, RobertaTokenizer
from tqdm import tqdm # 用于显示进度条

# ==============================================
# 0. 路径与配置 (请根据实际情况修改)
# ==============================================
model_path = "D:/model/MoralFoundationsClassifier"  
input_file = "inputters/data/data.jsonl"                      
output_file = "analysis/MORAL_FOUNDATIONS/moral_foundations.jsonl"

# 标签名称（英文）
label_names = [
    "care_virtue",      # Compassion, kindness, nurturing
    "care_vice",        # Harm, cruelty, suffering
    "fairness_virtue",  # Justice, equality, reciprocity
    "fairness_vice",    # Cheating, inequality, injustice
    "loyalty_virtue",   # Loyalty, patriotism, self-sacrifice
    "loyalty_vice",     # Betrayal, treason, disloyalty
    "authority_virtue", # Respect, tradition, order
    "authority_vice",   # Subversion, disobedience, chaos
    "sanctity_virtue",  # Purity, sanctity, nobility
    "sanctity_vice"     # Degradation, contamination, impurity
]

# ==============================================
# 1. 加载模型和分词器
# ==============================================
print("正在加载模型和分词器...")
tokenizer = RobertaTokenizer.from_pretrained(model_path)
model = RobertaForSequenceClassification.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
print(f"模型已加载到设备: {device}")

# ==============================================
# 2. 定义预测函数
# ==============================================
def predict_moral_foundations(text):
    """
    对输入文本进行 tokenization 并预测道德基础得分
    """
    # 截断和 padding，RoBERTa 最大长度通常为 512
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        
        # 道德基础通常是多标签分类，使用 sigmoid 计算概率
        # 如果你的模型是多分类单标签（Softmax），请将 torch.sigmoid 改为 torch.softmax(logits, dim=-1)
        probabilities = torch.sigmoid(logits).squeeze().cpu().tolist()
        
    # 将概率与标签名对应组合成字典，保留4位小数
    scores = {label: round(prob, 4) for label, prob in zip(label_names, probabilities)}
    return scores

# ==============================================
# 3. 处理数据并预测
# ==============================================
print("开始处理数据并进行预测...")

results = []

with open(input_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

# 使用 tqdm 显示处理进度
for line in tqdm(lines, desc="Predicting"):
    if not line.strip():
        continue
        
    try:
        # 注意：修复了你示例数据末尾多余的 ']' 问题
        data = json.loads(line.strip())
    except json.JSONDecodeError:
        print("JSON 解析错误，跳过该行。")
        continue

    dialogue_id = data.get("dialogue_id")
    utterances = data.get("utterances", [])
    binary_label = data.get("dialog_binary_label") # 1:道德绑架类, 0:非道德绑架类
    multi_label = data.get("dialog_multi_label")
    
    full_dialogue_text = ""
    utterance_analysis = []
    
    # --- 角度 1: 针对每一轮对话进行预测 ---
    for i, utt in enumerate(utterances):
        # 说话人交替往复
        speaker = "Person1" if i % 2 == 0 else "Person2"
        utt_with_speaker = f"{speaker}: {utt}"
        full_dialogue_text += f"{utt_with_speaker} "
        
        # 预测单轮句子
        utt_scores = predict_moral_foundations(utt_with_speaker)
        utterance_analysis.append({
            "turn": i + 1,
            "speaker": speaker,
            "text": utt,
            "moral_scores": utt_scores
        })
        
    # --- 角度 2: 针对整体对话进行预测 ---
    # 去除末尾多余的空格
    full_dialogue_text = full_dialogue_text.strip()
    dialogue_scores = predict_moral_foundations(full_dialogue_text)
    
    # --- 组装最终结果 ---
    # 我们保留了 dialog_binary_label，方便你后续区分类别
    result_data = {
        "dialogue_id": dialogue_id,
        "dialog_binary_label": binary_label,
        "dialog_multi_label": multi_label,
        "analysis": {
            "overall_dialogue_scores": dialogue_scores,
            "turn_by_turn_scores": utterance_analysis
        }
    }
    results.append(result_data)

# ==============================================
# 4. 保存预测结果到新的 JSONL 文件
# ==============================================
import os
os.makedirs(os.path.dirname(output_file), exist_ok=True)

with open(output_file, "w", encoding="utf-8") as f_out:
    for res in results:
        f_out.write(json.dumps(res, ensure_ascii=False) + "\n")

print(f"预测完成！结果已保存至: {output_file}")
print("后续你可以通过 'dialog_binary_label' 字段轻松将 1（道德绑架）和 0（非道德绑架）分开分析。")