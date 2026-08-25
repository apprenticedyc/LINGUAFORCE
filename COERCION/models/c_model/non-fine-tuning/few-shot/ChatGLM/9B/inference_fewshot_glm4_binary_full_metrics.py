import json
import random
import numpy as np
import torch
import re
# import os
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from transformers import AutoModelForCausalLM, AutoTokenizer

# ================= 1. 全局配置 =================
MODEL_NAME = "/root/autodl-tmp/glm-4-9b-chat" 
DATA_FILE = "/root/autodl-tmp/DATA/test.jsonl"
OUTPUT_FILE = "/root/autodl-tmp/non-fine-tuning/ChatGLM/few_shot_full_metrics.jsonl"

# 【新增】正负样本池路径
POSITIVE_SAMPLE_FILE = "/root/autodl-tmp/DATA/sample_positive.jsonl"
NEGATIVE_SAMPLE_FILE = "/root/autodl-tmp/DATA/sample_negative.jsonl"

# 实验变量
# 注意：现在的 K 代表"每个类别的样本数"。
# 例如 K=1 代表 1个正样本 + 1个负样本 (共2个)；K=0 代表 0-shot
SHOT_LIST = [0, 1, 2, 3, 4, 5, 6]
SEEDS = [42, 519, 2026]
MAX_TEST_SAMPLES = 99999  # 调试时改小

# ================= 2. 数据加载工具 =================

def load_jsonl(file_path):
    """通用 jsonl 加载函数"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    return data

# ================= 3. 核心算法 =================

def get_paired_shots(pos_pool, neg_pool, k, seed):
    """
    严格成对采样：每个 shot 包含 k 个正样本和 k 个负样本
    """
    if k == 0:
        return []
    
    rng = random.Random(seed)
    
    # 从两个池子中分别随机抽取 k 个样本
    # 注意：需确保 k 不超过池子的总数量
    sampled_pos = rng.sample(pos_pool, k)
    sampled_neg = rng.sample(neg_pool, k)
    
    # 合并后打乱顺序，防止模型产生位置依赖（比如总是最后看到正样本）
    shots = sampled_pos + sampled_neg
    rng.shuffle(shots)
    
    return shots

def format_dialogue(utterance_list):
    valid_utts = [u for u in utterance_list if u.strip()]
    return "\n".join([f"Utterance{i+1}: {u}" for i, u in enumerate(valid_utts)])

def build_prompt(dialogue_str):
    return (f"Below is a conversation. Please determine whether moral blackmail exists. "
            f"Output 1 if it exists; output 0 if it does not.\n"
            f"Conversation:\n{dialogue_str}\n\n"
            f"Your judgment (output only the number 0 or 1):")

# ================= 4. 主流程 =================

print("Loading data pools...")
pos_samples = load_jsonl(POSITIVE_SAMPLE_FILE)
neg_samples = load_jsonl(NEGATIVE_SAMPLE_FILE)
print(f"Loaded {len(pos_samples)} positive and {len(neg_samples)} negative samples for the few-shot pool.")

test_data = load_jsonl(DATA_FILE)
if len(test_data) > MAX_TEST_SAMPLES:
    test_data = test_data[:MAX_TEST_SAMPLES]
print(f"Loaded {len(test_data)} test samples.")


print(f"\nLoading GLM-4 model from {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, 
    torch_dtype="auto", 
    device_map="auto",
    trust_remote_code=True 
)

final_stats = {k: {'acc': [], 'prec': [], 'rec': [], 'f1': []} for k in SHOT_LIST}

for k in SHOT_LIST:
    print(f"\n{'='*20} Testing {k}-Shot per class (Total {k*2} examples) {'='*20}")
    
    for seed in SEEDS:
        # 1. 执行严格 1:1 采样
        current_shots = get_paired_shots(pos_samples, neg_samples, k=k, seed=seed)
        
        # 2. 构造 Base Prompt
        base_messages = [{"role": "system", "content": "You are a helpful assistant."}]
        for shot in current_shots:
            shot_text = format_dialogue(shot.get("utts", shot.get("utts", []))) 
            base_messages.append({"role": "user", "content": build_prompt(shot_text)})
            base_messages.append({"role": "assistant", "content": str(shot["label_b"])})
            
        y_true = []
        y_pred = []
        
        # 3. 批量推理循环
        for item in test_data:
            target_text = format_dialogue(item["utterances"])
            
            messages = base_messages.copy()
            messages.append({"role": "user", "content": build_prompt(target_text)})
            
            print(messages)

            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
                return_dict=True
            ).to(model.device)
            
            # 4. 生成
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=5, 
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            
            nums = re.findall(r'\d+', response)
            pred = int(nums[0]) if nums else 0
            
            y_true.append(item["dialog_binary_label"])
            y_pred.append(pred)
            
        # 4. 计算指标
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
        acc = accuracy_score(y_true, y_pred)
        
        final_stats[k]['acc'].append(acc)
        final_stats[k]['prec'].append(prec)
        final_stats[k]['rec'].append(rec)
        final_stats[k]['f1'].append(f1)
        
        print(f"  -> Seed {seed} | Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f}")

# ================= 5. 最终报告 =================

print("\n\n" + "="*85)
print(f"{'FINAL EXPERIMENT REPORT (GLM-4)':^85}")
print("="*85)
header = f"{'K (Per Class)':<13} | {'Acc (Mean±Std)':<16} | {'Prec (Mean±Std)':<16} | {'Rec (Mean±Std)':<16} | {'F1 (Mean±Std)':<16}"
print(header)
print("-" * 85)

results_for_file = []

for k in SHOT_LIST:
    accs = final_stats[k]['acc']
    precs = final_stats[k]['prec']
    recs = final_stats[k]['rec']
    f1s = final_stats[k]['f1']
    
    m_acc, s_acc = np.mean(accs)*100, np.std(accs)*100
    m_prec, s_prec = np.mean(precs)*100, np.std(precs)*100
    m_rec, s_rec = np.mean(recs)*100, np.std(recs)*100
    m_f1, s_f1 = np.mean(f1s)*100, np.std(f1s)*100
    
    row = (f"{k:<13} | "
           f"{m_acc:.1f}±{s_acc:.1f}%".ljust(16) + " | "
           f"{m_prec:.1f}±{s_prec:.1f}%".ljust(16) + " | "
           f"{m_rec:.1f}±{s_rec:.1f}%".ljust(16) + " | "
           f"{m_f1:.1f}±{s_f1:.1f}%".ljust(16))
    print(row)
    
    results_for_file.append({
        "k_shot_per_class": k,
        "total_shots_in_prompt": k * 2,
        "metrics": {
            "accuracy": f"{m_acc:.2f}±{s_acc:.2f}",
            "precision": f"{m_prec:.2f}±{s_prec:.2f}",
            "recall": f"{m_rec:.2f}±{s_rec:.2f}",
            "f1": f"{m_f1:.2f}±{s_f1:.2f}"
        },
        "raw_data": final_stats[k]
    })

with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    for r in results_for_file:
        f.write(json.dumps(r) + "\n")

print("="*85)
print(f"Detailed metrics saved to {OUTPUT_FILE}")