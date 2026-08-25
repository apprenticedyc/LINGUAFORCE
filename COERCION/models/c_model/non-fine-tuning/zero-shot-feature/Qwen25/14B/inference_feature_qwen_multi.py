import json
import re
import torch
import itertools
import os
import random
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# ================= 1. 配置区域 (Configuration) =================
MODEL_PATH = "/root/autodl-tmp/Qwen25-14B-Instruct" 
DATA_FILE = "/root/autodl-tmp/DATA/updated_test.jsonl"
BASE_OUTPUT_DIR = "/root/autodl-tmp/non-fine-tuning/Qwen25/multi"
SUMMARY_FILE = os.path.join(BASE_OUTPUT_DIR, "final_ablation_summary.json")

# 随机种子：增加实验结果的鲁棒性
SEEDS = [42, 520, 2026]

# 粒度消融模式
GRANULARITY_MODES = ["global_only", "turn_only", "both"]

# 最大测试样本数 (None 表示跑全量)
MAX_SAMPLES = None 

# --- 特征定义库 ---
FEATURE_DEFINITIONS = {
    0: ("Obligation", "Measures pressure regarding duty. 'Low': Gentle suggestions. 'Moderate': Explicit pressure using social/family roles. 'High': Intense, unavoidable demands."),
    1: ("Constraint", "Measures restriction of choice. 'Low': Subtle narrowing. 'Moderate': Conditional threats ('if/then') or logical traps. 'High': Total deprivation of choice."),
    2: ("ValueJudgement", "Measures moral labels. 'Low': Subjective preferences. 'Moderate': Loaded moral terms (e.g., 'selfish', 'ungrateful'). 'High': Severe character assassination."),
    3: ("Toxicity", "Measures aggressive tone. 'Low': Passive-aggressive or sarcastic. 'Moderate': Overtly rude. 'High': Hateful or abusive language.")
}
FEATURE_NAMES = {0: "Obligation", 1: "Constraint", 2: "ValueJudgement", 3: "Toxicity"}

# ================= 2. 辅助函数 =================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def parse_model_output(response):
    """解析 0-5 的整数输出"""
    response = response.strip()
    if response in [str(i) for i in range(6)]:
        return int(response)
    nums = re.findall(r'\b[0-5]\b', response)
    if nums:
        return int(nums[0])
    return -1

def map_to_binary(score):
    """映射逻辑：0, 1 -> 0 (负类); 2, 3, 4, 5 -> 1 (正类)"""
    if score == -1: return -1
    return 0 if score <= 1 else 1

def get_feature_display_str(val_dict):
    if not isinstance(val_dict, dict):
        return None
    level = val_dict.get("level", "None")
    score = val_dict.get("intensity_score", 0.0)
    if level != "None" or score > 0.05:
        return f"{level} (Intensity: {score:.2f})"
    return None

# ================= 3. 动态 Prompt 生成 =================

def generate_prompt_content(item, active_indices, granularity):
    # A. 特征定义
    definitions = [f"- {FEATURE_DEFINITIONS[idx][0]}: {FEATURE_DEFINITIONS[idx][1]}" for idx in active_indices]
    def_block = "\n".join(definitions)

    # B. Global 特征
    global_block = ""
    if granularity in ["global_only", "both"]:
        lines = []
        overall = item.get("overall_analysis", {}) 
        for idx in active_indices:
            name_key = FEATURE_NAMES[idx]
            if name_key in overall:
                feat_str = get_feature_display_str(overall[name_key])
                if feat_str: lines.append(f"- {name_key}: {feat_str}")
        
        # FIX 1: Prevent backslash in f-string or string concatenation issues here
        global_lines_str = "\n".join(lines) if lines else "None detected."
        global_block = f"Global Analysis Signals:\n{global_lines_str}\n"

    # C. Turn-level 特征与对话
    utterances = item.get("utts", item.get("utterances", []))
    turn_analysis = item.get("turn_analysis", [])
    structured_turn_map = {t['turn_id']: t.get('features', {}) for t in turn_analysis}
    
    dialogue_lines = []
    for i, text in enumerate(utterances):
        line = f"Turn {i}: \"{text}\""
        if granularity in ["turn_only", "both"]:
            turn_signals = []
            for idx in active_indices:
                name_key = FEATURE_NAMES[idx]
                if i in structured_turn_map and name_key in structured_turn_map[i]:
                    feat_str = get_feature_display_str(structured_turn_map[i][name_key])
                    if feat_str: turn_signals.append(f"{name_key}: {feat_str}")
            if turn_signals: line += f"\n   [Signals: {', '.join(turn_signals)}]"
        dialogue_lines.append(line)
    
    # FIX 2: Create the text variable before the f-string
    dialogue_text = "\n".join(dialogue_lines)
    
    # D. 最终组合
    prompt = f"""### Role
You are an expert linguist specializing in pragmatics and moral coercion detection.

### Task
Analyze the dialogue to determine the severity of moral coercion (Scale 0-5).
Use the provided linguistic feature signals as expert hints.

### Feature Definitions
{def_block}

### Input Data
{global_block}
Dialogue Transcript:
{dialogue_text}

### Output Requirement
Assess the severity of moral coercion on a scale from 0 to 5.
(0 indicates no moral coercion, 5 indicates extreme moral coercion)

Scale Definitions:
0: No moral coercion
1: Slight suspicion (Ambiguous)
2: Slight tendency (Weak)
3: Clear implication (Moderate)
4: Highly coercive (Strong)
5: Extreme moral coercion (Severe)

Your Answer (output only a single number between 0 and 5):
"""
    return prompt

# ================= 4. 主程序推理与评估 =================

if __name__ == "__main__":
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    
    print(f"Loading model and data...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto")

    all_data = []
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            try: all_data.append(json.loads(line.strip()))
            except: continue
    if MAX_SAMPLES: all_data = all_data[:MAX_SAMPLES]

    # 生成所有特征组合 (1-4个特征)
    feat_combinations = []
    for r in range(1, 5): 
        feat_combinations.extend(itertools.combinations(range(4), r))
    
    final_summary = []

    for granularity in GRANULARITY_MODES:
        for combo in feat_combinations:
            combo_key = "+".join([FEATURE_NAMES[idx] for idx in combo])
            print(f"\n>>> Running: {granularity} | {combo_key}")
            
            # 指标容器
            seed_metrics = {
                "m_acc": [], "m_prec": [], "m_rec": [], "m_f1": [],
                "b_acc": [], "b_prec": [], "b_rec": [], "b_f1": []
            }
            
            for seed in SEEDS:
                set_seed(seed)
                seed_dir = os.path.join(BASE_OUTPUT_DIR, f"seed_{seed}")
                os.makedirs(seed_dir, exist_ok=True)
                
                y_true_multi, y_pred_multi = [], []
                results_to_save = []

                for item in tqdm(all_data, desc=f"Seed {seed}", leave=False):
                    prompt = generate_prompt_content(item, combo, granularity)
                    # print(prompt)
                    inputs = tokenizer([tokenizer.apply_chat_template([{"role":"user","content":prompt}], tokenize=False, add_generation_prompt=True)], return_tensors="pt").to(model.device)
                    
                    with torch.no_grad():
                        output_ids = model.generate(**inputs, max_new_tokens=5, do_sample=True, temperature=0.7, pad_token_id=tokenizer.eos_token_id)
                    
                    response = tokenizer.decode(output_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                    pred = parse_model_output(response)
                    true = item.get("label_m", item.get("dialog_multi_label", 0))

                    if pred != -1:
                        y_true_multi.append(true)
                        y_pred_multi.append(pred)
                    
                    results_to_save.append({"id": item.get("dialogue_id"), "true": true, "pred": pred, "raw": response})

                # 计算该 Seed 的指标
                if y_true_multi:
                    # 6分类指标 (Macro)
                    ma = accuracy_score(y_true_multi, y_pred_multi)
                    mp, mr, mf, _ = precision_recall_fscore_support(y_true_multi, y_pred_multi, average='macro', zero_division=0)
                    seed_metrics["m_acc"].append(ma); seed_metrics["m_prec"].append(mp); seed_metrics["m_rec"].append(mr); seed_metrics["m_f1"].append(mf)

                    # 二分类指标
                    y_true_bin = [map_to_binary(x) for x in y_true_multi]
                    y_pred_bin = [map_to_binary(x) for x in y_pred_multi]
                    ba = accuracy_score(y_true_bin, y_pred_bin)
                    bp, br, bf, _ = precision_recall_fscore_support(y_true_bin, y_pred_bin, average='binary', zero_division=0)
                    seed_metrics["b_acc"].append(ba); seed_metrics["b_prec"].append(bp); seed_metrics["b_rec"].append(br); seed_metrics["b_f1"].append(bf)

                # 保存该 Seed 的原始预测
                with open(os.path.join(seed_dir, f"{granularity}_{combo_key.replace(' ','')}.jsonl"), 'w') as f:
                    for r in results_to_save: f.write(json.dumps(r) + "\n")

            # 汇总多 Seed 结果
            if seed_metrics["m_f1"]:
                res = {
                    "granularity": granularity, 
                    "features": combo_key,
                    "multi": {
                        "acc_mean": np.mean(seed_metrics["m_acc"]), "acc_std": np.std(seed_metrics["m_acc"]),
                        "prec_mean": np.mean(seed_metrics["m_prec"]), "prec_std": np.std(seed_metrics["m_prec"]),
                        "rec_mean": np.mean(seed_metrics["m_rec"]), "rec_std": np.std(seed_metrics["m_rec"]),
                        "f1_mean": np.mean(seed_metrics["m_f1"]), "f1_std": np.std(seed_metrics["m_f1"]),
                    },
                    "binary": {
                        "acc_mean": np.mean(seed_metrics["b_acc"]), "acc_std": np.std(seed_metrics["b_acc"]),
                        "prec_mean": np.mean(seed_metrics["b_prec"]), "prec_std": np.std(seed_metrics["b_prec"]),
                        "rec_mean": np.mean(seed_metrics["b_rec"]), "rec_std": np.std(seed_metrics["b_rec"]),
                        "f1_mean": np.mean(seed_metrics["b_f1"]), "f1_std": np.std(seed_metrics["b_f1"]),
                    }
                }
                final_summary.append(res)

    # ================= 5. 排序与输出表格 =================
    # 按二分类的 f1_mean 降序排序
    final_summary.sort(key=lambda x: x['binary']['f1_mean'], reverse=True)
    
    # 打印排版好的表格
    print("\n" + "=" * 165)
    header = f"{'Mode':<12} | {'Features':<30} | {'B-Acc':<13} | {'B-Pre':<13} | {'B-Rec':<13} | {'B-F1':<13} | {'M-Acc':<13} | {'M-Pre':<13} | {'M-Rec':<13} | {'M-F1':<13}"
    print(header)
    print("-" * 165)
    
    for item in final_summary:
        f, g = item['features'], item['granularity']
        bm, mm = item['binary'], item['multi']
        
        b_acc = f"{bm['acc_mean']:.4f}±{bm['acc_std']:.4f}"
        b_pre = f"{bm['prec_mean']:.4f}±{bm['prec_std']:.4f}"
        b_rec = f"{bm['rec_mean']:.4f}±{bm['rec_std']:.4f}"
        b_f1  = f"{bm['f1_mean']:.4f}±{bm['f1_std']:.4f}"
        
        m_acc = f"{mm['acc_mean']:.4f}±{mm['acc_std']:.4f}"
        m_pre = f"{mm['prec_mean']:.4f}±{mm['prec_std']:.4f}"
        m_rec = f"{mm['rec_mean']:.4f}±{mm['rec_std']:.4f}"
        m_f1  = f"{mm['f1_mean']:.4f}±{mm['f1_std']:.4f}"
        
        row = f"{g:<12} | {f:<30} | {b_acc:<13} | {b_pre:<13} | {b_rec:<13} | {b_f1:<13} | {m_acc:<13} | {m_pre:<13} | {m_rec:<13} | {m_f1:<13}"
        print(row)
        
    print("=" * 165)

    with open(SUMMARY_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_summary, f, indent=4, ensure_ascii=False)