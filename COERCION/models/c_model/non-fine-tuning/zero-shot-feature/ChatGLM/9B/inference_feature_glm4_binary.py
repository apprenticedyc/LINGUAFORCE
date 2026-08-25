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
# 【修改点 1】模型路径和输出路径改为 GLM-4
MODEL_PATH = "/root/autodl-tmp/glm-4-9b-chat" 
DATA_FILE = "/root/autodl-tmp/DATA/updated_test.jsonl"
BASE_OUTPUT_DIR = "/root/autodl-tmp/non-fine-tuning/ChatGLM/binary"
SUMMARY_FILE = os.path.join(BASE_OUTPUT_DIR, "final_ablation_summary.json")

# 随机种子
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
    response = response.strip()
    if response in ["0", "1"]:
        return int(response)
    nums = re.findall(r'\b[01]\b', response)
    if nums:
        return int(nums[0])
    return -1

def get_feature_display_str(val_dict):
    if not isinstance(val_dict, dict):
        return None
        
    level = val_dict.get("level", "None")
    score = val_dict.get("intensity_score", 0.0)
    
    if level != "None" or score > 0.05:
        return f"{level} (Intensity: {score:.2f})"
    return None

# ================= 3. 核心逻辑：动态 Prompt 生成 =================

def generate_prompt_content(item, active_indices, granularity):
    # --- A. 生成定义部分 ---
    definitions = []
    for idx in active_indices:
        name, desc = FEATURE_DEFINITIONS[idx]
        definitions.append(f"- {name}: {desc}")
    def_block = "\n".join(definitions)

    # --- B. 生成 Global 特征块 ---
    global_block = ""
    if granularity in ["global_only", "both"]:
        lines = []
        overall = item.get("overall_analysis", {})

        for idx in active_indices:
            name_key = FEATURE_NAMES[idx]
            if name_key in overall:
                feat_str = get_feature_display_str(overall[name_key])
                if feat_str:
                    lines.append(f"- {name_key}: {feat_str}")

        if lines:
            global_block = "Global Analysis Signals:\n" + "\n".join(lines) + "\n"
        else:
            global_block = "Global Analysis Signals: None detected.\n"

    # --- C. 生成 Turn-level 特征块 ---
    utterances = item.get("utterances", [])
    dialogue_lines = []
    
    turn_analysis = item.get("turn_analysis", [])
    structured_turn_map = {t['turn_id']: t.get('features', {}) for t in turn_analysis}

    for i, text in enumerate(utterances):
        line = f"Turn {i}: \"{text}\""
        
        if granularity in ["turn_only", "both"]:
            turn_signals = []
            for idx in active_indices:
                name_key = FEATURE_NAMES[idx]
                if i in structured_turn_map and name_key in structured_turn_map[i]:
                    feat_str = get_feature_display_str(structured_turn_map[i][name_key])
                    if feat_str:
                        turn_signals.append(f"{name_key}: {feat_str}")
            
            if turn_signals:
                line += f"\n   [Signals: {', '.join(turn_signals)}]"
        
        dialogue_lines.append(line)
    
    dialogue_text = "\n".join(dialogue_lines)

    # --- D. 组装最终 Prompt ---
    prompt = f"""### Role
You are an expert linguist specializing in pragmatics and moral coercion detection.

### Task
Analyze the dialogue to determine if this is a case of moral coercion.
Use the provided linguistic feature signals as expert hints.

### Feature Definitions
{def_block}

### Input Data
{global_block}
Dialogue Transcript:
{dialogue_text}

### Output Requirement
Determine if this is a case of moral coercion.
Your Answer 1 if YES, 0 if NO (Number only).
"""
    return prompt

if __name__ == "__main__":
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    
    print(f"Loading GLM-4 model from: {MODEL_PATH}")
    # 【修改点 2】GLM-4 必须加 trust_remote_code=True
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, 
        trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, 
        torch_dtype=torch.bfloat16, 
        device_map="auto",
        trust_remote_code=True
    )

    print(f"Loading data: {DATA_FILE}")
    all_data = []
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                all_data.append(json.loads(line.strip()))
    
    if MAX_SAMPLES:
        all_data = all_data[:MAX_SAMPLES]
        print(f"Debug: Truncated to {MAX_SAMPLES} samples.")

    feature_indices = [0, 1, 2, 3]
    feat_combinations = []
    for r in range(1, 5): 
        feat_combinations.extend(itertools.combinations(feature_indices, r))
    
    print(f"Total Combinations: {len(feat_combinations)} | Granularities: {len(GRANULARITY_MODES)}")
    
    final_summary = []

    # ================= 5. 双重循环消融实验 =================
    
    for granularity in GRANULARITY_MODES:
        print(f"\n{'#'*60}")
        print(f"Running Mode: {granularity.upper()}")
        print(f"{'#'*60}")
        
        for combo in feat_combinations:
            combo_names = [FEATURE_NAMES[idx] for idx in combo]
            combo_key = "+".join(combo_names)
            safe_filename_key = combo_key.replace(" ", "")
            
            print(f"--- Combo: {combo_key} ---")
            
            metrics_buffer = {"acc": [], "f1": [], "prec": [], "rec": []}
            
            for seed in SEEDS:
                set_seed(seed)
                seed_dir = os.path.join(BASE_OUTPUT_DIR, f"seed_{seed}")
                os.makedirs(seed_dir, exist_ok=True)
                output_file = os.path.join(seed_dir, f"res_{granularity}_{safe_filename_key}.jsonl")
                
                results = []
                
                for item in tqdm(all_data, desc=f"Seed {seed}", leave=False):
                    prompt_text = generate_prompt_content(item, combo, granularity)
                    
                    messages = [
                        {"role": "system", "content": "You are an expert linguist specializing in pragmatics."},
                        {"role": "user", "content": prompt_text}
                    ]
                    
                    # 【修改点 3】GLM-4 推荐的推理逻辑，直接获取 tensor 字典
                    inputs = tokenizer.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        tokenize=True,
                        return_tensors="pt",
                        return_dict=True
                    ).to(model.device)
                    
                    input_len = inputs.input_ids.shape[1]
                    
                    with torch.no_grad():
                        generated_ids = model.generate(
                            **inputs, # 展开 inputs 字典
                            max_new_tokens=5,
                            do_sample=True,
                            temperature=0.7,
                            top_p=0.9,
                            pad_token_id=tokenizer.eos_token_id,
                            eos_token_id=tokenizer.eos_token_id
                        )
                    
                    # 解码输出
                    output_ids = generated_ids[:, input_len:]
                    response = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
                    
                    pred = parse_model_output(response)
                    
                    results.append({
                        "id": item.get("dialogue_id"),
                        "true": item.get("dialog_binary_label", -1),
                        "pred": pred,
                        "raw": response
                    })
                
                # 保存单次推理结果
                with open(output_file, 'w', encoding='utf-8') as f:
                    for r in results:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
                
                valid_res = [r for r in results if r["pred"] != -1]
                if valid_res:
                    y_true = [r["true"] for r in valid_res]
                    y_pred = [r["pred"] for r in valid_res]
                    acc = accuracy_score(y_true, y_pred)
                    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
                    metrics_buffer["acc"].append(acc)
                    metrics_buffer["f1"].append(f1)
                    metrics_buffer["prec"].append(p)
                    metrics_buffer["rec"].append(r)

                # 【修改点 4】清理显存防碎片化
                torch.cuda.empty_cache()
            
            if metrics_buffer["f1"]:
                mean_f1 = np.mean(metrics_buffer["f1"])
                std_f1 = np.std(metrics_buffer["f1"])
                mean_acc = np.mean(metrics_buffer["acc"])
                std_acc = np.std(metrics_buffer["acc"])
                
                print(f"Result: F1 = {mean_f1:.4f} ± {std_f1:.4f} | Acc = {mean_acc:.4f}")
                
                final_summary.append({
                    "granularity": granularity,
                    "features": combo_key,
                    "metrics": {
                        "f1_mean": mean_f1, "f1_std": std_f1,
                        "acc_mean": mean_acc, "acc_std": std_acc,
                        "prec_mean": np.mean(metrics_buffer["prec"]),
                        "rec_mean": np.mean(metrics_buffer["rec"])
                    }
                })

    # ================= 6. 保存总结 =================
    final_summary.sort(key=lambda x: x['metrics']['f1_mean'], reverse=True)
    
    print("\n" + "="*80)
    print("FINAL ABLATION LEADERBOARD")
    print("="*80)
    print(f"{'Mode':<12} | {'Features':<40} | {'F1':<10} | {'Acc':<10}")
    print("-" * 75)
    for item in final_summary:
        print(f"{item['granularity']:<12} | {item['features']:<40} | {item['metrics']['f1_mean']:.4f}     | {item['metrics']['acc_mean']:.4f}")
    
    with open(SUMMARY_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_summary, f, indent=4, ensure_ascii=False)

    print(f"\nAll done! Summary saved to {SUMMARY_FILE}")