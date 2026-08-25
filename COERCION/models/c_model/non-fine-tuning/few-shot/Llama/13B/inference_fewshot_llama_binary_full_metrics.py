import os
import json
import random
import numpy as np
import collections
import re
import torch
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from transformers import AutoModelForCausalLM, AutoTokenizer

# ================= 1. 全局配置 =================
MODEL_NAME = "/root/autodl-tmp/Llama-2-13b-chat-ms" 
DATA_FILE = "/root/autodl-tmp/DATA/test.jsonl"
OUTPUT_BASE_DIR = "/root/autodl-tmp/non-fine-tuning/Llama2"
SUMMARY_FILE = os.path.join(OUTPUT_BASE_DIR, "few_shot_full_metrics.jsonl")

# 【新增】外部样本池路径
POSITIVE_SAMPLE_FILE = "/root/autodl-tmp/DATA/sample_positive.jsonl"
NEGATIVE_SAMPLE_FILE = "/root/autodl-tmp/DATA/sample_negative.jsonl"

# 实验变量
# k=1 表示 1正1负（共2个）；k=0 为 Zero-shot
SHOT_LIST = [0, 1, 2, 3, 4, 5, 6] 
SEEDS = [42, 519, 2026] 
MAX_TEST_SAMPLES = 99999 

# ================= 2. 初始化环境与数据加载 =================
os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
for seed in SEEDS:
    os.makedirs(os.path.join(OUTPUT_BASE_DIR, f"seed_{seed}"), exist_ok=True)

def load_jsonl(file_path):
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
    
    # 无放回采样
    sampled_pos = rng.sample(pos_pool, k)
    sampled_neg = rng.sample(neg_pool, k)
    
    # 合并并打乱顺序
    shots = sampled_pos + sampled_neg
    rng.shuffle(shots)
    return shots

def format_dialogue(utterance_list):
    valid_utts = [u for u in utterance_list if u.strip()]
    return "\n".join([f"Utterance{i+1}: {u}" for i, u in enumerate(valid_utts)])

# ================= 4. 主流程 =================

print("Loading data pools...")
pos_samples = load_jsonl(POSITIVE_SAMPLE_FILE)
neg_samples = load_jsonl(NEGATIVE_SAMPLE_FILE)

print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, 
    torch_dtype=torch.float16, 
    device_map="auto"
)

test_data = load_jsonl(DATA_FILE)
if len(test_data) > MAX_TEST_SAMPLES:
    test_data = test_data[:MAX_TEST_SAMPLES]

final_stats = {k: {'acc': [], 'prec': [], 'rec': [], 'f1': []} for k in SHOT_LIST}

for k in SHOT_LIST:
    print(f"\n{'='*20} Testing {k}-Shot Pairs (Total {k*2} examples) {'='*20}")
    
    for seed in SEEDS:
        # 获取成对样本
        current_shots = get_paired_shots(pos_samples, neg_samples, k=k, seed=seed)
        
        y_true = []
        y_pred = []
        current_run_predictions = [] 
        
        for item in test_data:
            target_text = format_dialogue(item["utterances"])
            
            # --- 构建 Raw Prompt (引导补全模式) ---
            raw_prompt = "Below is a conversation. Please determine whether moral blackmail exists. Output strictly 1 for Yes, or 0 for No.\n\n"
            
            # 拼接 Few-shot 示例
            for shot in current_shots:
                # 兼容不同字段名
                shot_utts = shot.get("utts", shot.get("utterances", []))
                shot_text = format_dialogue(shot_utts)
                raw_prompt += f"Conversation:\n{shot_text}\nOutput: {shot['label_b']}\n\n"
            
            # 拼接最终测试目标
            raw_prompt += f"Conversation:\n{target_text}\nOutput:"
            
            # --- 编码与推理 ---
            inputs = tokenizer(raw_prompt, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=5, 
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
                
            response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
            
            # 提取数字
            match = re.search(r'\b(0|1)\b', response)
            pred = int(match.group(1)) if match else 0
            
            y_true.append(item["dialog_binary_label"])
            y_pred.append(pred)
            
            current_run_predictions.append({
                "dialogue_id": item.get("id", "unknown"),
                "true_binary": item["dialog_binary_label"],
                "pred_binary": pred,
                "model_response": response.strip()
            })
            
        # 计算指标
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
        acc = accuracy_score(y_true, y_pred)
        
        final_stats[k]['acc'].append(acc)
        final_stats[k]['prec'].append(prec)
        final_stats[k]['rec'].append(rec)
        final_stats[k]['f1'].append(f1)
        
        # 实时保存当前 Seed 的结果
        shot_file_path = os.path.join(OUTPUT_BASE_DIR, f"seed_{seed}", f"{k}_shot.jsonl")
        with open(shot_file_path, 'w', encoding='utf-8') as pf:
            for record in current_run_predictions:
                pf.write(json.dumps(record) + "\n")
                
        print(f"  -> Seed {seed} | Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f}")

# ================= 5. 输出汇总报告 =================
print("\n\n" + "="*85)
print(f"{'FINAL EXPERIMENT REPORT (Llama-2-13b)':^85}")
print("="*85)
header = f"{'K (Pairs)':<10} | {'Acc (Mean±Std)':<18} | {'Prec (Mean±Std)':<18} | {'Rec (Mean±Std)':<18} | {'F1 (Mean±Std)':<18}"
print(header)
print("-" * 85)

results_for_file = []
for k in SHOT_LIST:
    accs, precs, recs, f1s = final_stats[k]['acc'], final_stats[k]['prec'], final_stats[k]['rec'], final_stats[k]['f1']
    m_acc, s_acc = np.mean(accs)*100, np.std(accs)*100
    m_prec, s_prec = np.mean(precs)*100, np.std(precs)*100
    m_rec, s_rec = np.mean(recs)*100, np.std(recs)*100
    m_f1, s_f1 = np.mean(f1s)*100, np.std(f1s)*100
    
    row = (f"{k:<10} | {m_acc:.1f}±{s_acc:.1f}%".ljust(21) + f" | {m_prec:.1f}±{s_prec:.1f}%".ljust(21) + 
           f" | {m_rec:.1f}±{s_rec:.1f}%".ljust(21) + f" | {m_f1:.1f}±{s_f1:.1f}%")
    print(row)
    
    results_for_file.append({
        "k_shot_pairs": k,
        "total_examples": k * 2,
        "metrics": {"accuracy": f"{m_acc:.2f}±{s_acc:.2f}", "f1": f"{m_f1:.2f}±{s_f1:.2f}"},
        "raw_data": final_stats[k]
    })

with open(SUMMARY_FILE, 'w', encoding='utf-8') as f:
    for r in results_for_file:
        f.write(json.dumps(r) + "\n")

print("="*85)
print(f"Summary saved to: {SUMMARY_FILE}")