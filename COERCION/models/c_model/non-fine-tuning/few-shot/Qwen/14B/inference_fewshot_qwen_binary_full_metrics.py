import json
import random
import numpy as np
import torch
import re
# import os
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from transformers import AutoModelForCausalLM, AutoTokenizer

# ================= 1. 全局配置 =================
MODEL_NAME = "/root/autodl-tmp/Qwen25-14B-Instruct"
DATA_FILE = "/root/autodl-tmp/DATA/test.jsonl"
OUTPUT_FILE = "/root/autodl-tmp/non-fine-tuning/Qwen25/few_shot_full_metrics.jsonl"

# 【新增】正负样本池路径
POSITIVE_SAMPLE_FILE = "/root/autodl-tmp/DATA/sample_positive.jsonl"
NEGATIVE_SAMPLE_FILE = "/root/autodl-tmp/DATA/sample_negative.jsonl"

# 实验变量
# SHOT_LIST 中的 k 代表"正负各取 k 个"。例如 k=1，总 Prompt 中会有 2 个案例。
SHOT_LIST = [0, 1, 2, 3, 4, 5, 6]
SEEDS = [42, 519, 2026]
MAX_TEST_SAMPLES = 99999 

# ================= 2. 数据加载工具 =================

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    return data

# ================= 3. 核心算法：成对采样 =================

def get_paired_shots(pos_pool, neg_pool, k, seed):
    """
    严格成对采样：从正样本池抽 k 个，从负样本池抽 k 个，混合并打乱。
    """
    if k == 0:
        return []
    
    rng = random.Random(seed)
    
    # 无放回采样 k 个 (random.sample 保证不重复)
    sampled_pos = rng.sample(pos_pool, k)
    sampled_neg = rng.sample(neg_pool, k)
    
    # 合并打乱，消除位置偏差
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
print(f"Loaded {len(pos_samples)} positive and {len(neg_samples)} negative samples.")

test_data = load_jsonl(DATA_FILE)
if len(test_data) > MAX_TEST_SAMPLES:
    test_data = test_data[:MAX_TEST_SAMPLES]
print(f"Loaded {len(test_data)} test samples.")

print(f"\nLoading Qwen2.5 model from {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, 
    torch_dtype="auto", 
    device_map="auto"
)

final_stats = {k: {'acc': [], 'prec': [], 'rec': [], 'f1': []} for k in SHOT_LIST}

for k in SHOT_LIST:
    print(f"\n{'='*20} Testing {k}-Shot Pair (Total {k*2} examples) {'='*20}")
    
    for seed in SEEDS:
        # 1. 执行成对采样
        current_shots = get_paired_shots(pos_samples, neg_samples, k=k, seed=seed)
        
        # 2. 构造 Base Prompt (Qwen 风格)
        base_messages = [{"role": "system", "content": "You are a helpful assistant."}]
        for shot in current_shots:
            # 兼容 utts 或 utterances 字段名
            shot_utts = shot.get("utts", shot.get("utterances", []))
            shot_text = format_dialogue(shot_utts)
            base_messages.append({"role": "user", "content": build_prompt(shot_text)})
            base_messages.append({"role": "assistant", "content": str(shot["label_b"])})
            
        y_true = []
        y_pred = []
        
        # 3. 循环测试
        for item in test_data:
            target_text = format_dialogue(item["utterances"])
            
            messages = base_messages.copy()
            messages.append({"role": "user", "content": build_prompt(target_text)})
            
            # 使用 Qwen 的 Chat Template 处理
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
            
            # 4. 生成 (贪心解码)
            with torch.no_grad():
                generated_ids = model.generate(
                    **model_inputs,
                    max_new_tokens=5,
                    do_sample=False,   # 必须设为 False 保证评估一致性
                    temperature=None,  # 设为 None 配合 do_sample=False
                    top_p=None,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # 剪切 input 部分，仅保留生成的 output
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # 5. 结果提取
            nums = re.findall(r'\d+', response)
            pred = int(nums[0]) if nums else 0
            
            y_true.append(item["dialog_binary_label"])
            y_pred.append(pred)
            
        # 6. 计算本 Seed 指标
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
        acc = accuracy_score(y_true, y_pred)
        
        final_stats[k]['acc'].append(acc)
        final_stats[k]['prec'].append(prec)
        final_stats[k]['rec'].append(rec)
        final_stats[k]['f1'].append(f1)
        
        print(f"  -> Seed {seed} | Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f}")

# ================= 5. 总结报告 =================

print("\n\n" + "="*85)
print(f"{'FINAL EXPERIMENT REPORT (Qwen2.5-14B)':^85}")
print("="*85)
header = f"{'K (Pairs)':<10} | {'Acc (Mean±Std)':<16} | {'Prec (Mean±Std)':<16} | {'Rec (Mean±Std)':<16} | {'F1 (Mean±Std)':<16}"
print(header)
print("-" * 85)

results_for_file = []

for k in SHOT_LIST:
    accs, precs, recs, f1s = final_stats[k]['acc'], final_stats[k]['prec'], final_stats[k]['rec'], final_stats[k]['f1']
    
    m_acc, s_acc = np.mean(accs)*100, np.std(accs)*100
    m_prec, s_prec = np.mean(precs)*100, np.std(precs)*100
    m_rec, s_rec = np.mean(recs)*100, np.std(recs)*100
    m_f1, s_f1 = np.mean(f1s)*100, np.std(f1s)*100
    
    row = (f"{k:<10} | "
           f"{m_acc:.1f}±{s_acc:.1f}%".ljust(16) + " | "
           f"{m_prec:.1f}±{s_prec:.1f}%".ljust(16) + " | "
           f"{m_rec:.1f}±{s_rec:.1f}%".ljust(16) + " | "
           f"{m_f1:.1f}±{s_f1:.1f}%".ljust(16))
    print(row)
    
    results_for_file.append({
        "k_shot_pairs": k,
        "total_examples": k * 2,
        "metrics_summary": {
            "accuracy": f"{m_acc:.2f}±{s_acc:.2f}",
            "f1_score": f"{m_f1:.2f}±{s_f1:.2f}"
        },
        "raw_results": final_stats[k]
    })

with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    for r in results_for_file:
        f.write(json.dumps(r) + "\n")

print("="*85)
print(f"Results successfully saved to: {OUTPUT_FILE}")