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
# 更新为 Llama-2 的路径
MODEL_PATH = "/root/autodl-tmp/Llama-2-13b-chat-ms" 
DATA_FILE = "/root/autodl-tmp/DATA/updated_test.jsonl"
BASE_OUTPUT_DIR = "/root/autodl-tmp/non-fine-tuning/Llama2_feature/binary_feature_ablation"
SUMMARY_FILE = os.path.join(BASE_OUTPUT_DIR, "final_ablation_summary.json")

# 随机种子
SEEDS = [42, 520, 2026]

# 粒度消融模式
GRANULARITY_MODES = ["global_only", "turn_only", "both"] 

# 最大测试样本数 (调试时设为 10，跑全量设为 None)
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
    # 严格抓取独立的 0 或 1，防止抓到特征定义里的数字
    match = re.search(r'\b(0|1)\b', response)
    if match:
        return int(match.group(1))
    return 0  # 默认兜底为 0

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
        # else:
        #     global_block = "Global Analysis Signals: None detected.\n"

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

    # --- D. 组装最终 Prompt (使用纯文本填空法) ---
    prompt = f"""Task: Analyze the dialogue to determine if this is a case of 'moral coercion' (moral blackmail).
Use the provided linguistic feature signals as expert hints.

### Feature Definitions
{def_block}

### Input Data
{global_block}
Dialogue Transcript:
{dialogue_text}

Question: Does this conversation contain moral coercion?
Please output strictly 1 for Yes, or 0 for No.
Output:"""
    return prompt

if __name__ == "__main__":
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    
    print(f"Loading Llama-2 model from: {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, 
        torch_dtype=torch.float16, # Llama-2 建议使用 float16
        device_map="auto"
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
                    # print(prompt_text)
                    # 抛弃 messages 格式，直接输入纯文本 prompt
                    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
                    
                    input_len = inputs.input_ids.shape[1]
                    
                    with torch.no_grad():
                        generated_ids = model.generate(
                            **inputs, 
                            max_new_tokens=5, # 填空题不需要很长的生成
                            do_sample=True,  # 分类任务关闭采样
                            pad_token_id=tokenizer.eos_token_id
                        )
                    
                    # 解码输出
                    output_ids = generated_ids[:, input_len:]
                    response = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
                    
                    pred = parse_model_output(response)
                    
                    results.append({
                        "id": item.get("dialogue_id", item.get("id")),
                        "true": item.get("dialog_binary_label", -1),
                        "pred": pred,
                        "raw": response # 记录原始回复以便 debug
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

                # 清理显存防碎片化
                torch.cuda.empty_cache()
            
            # --- 以下是替换的部分：全面计算 4 个指标的 Mean 和 Std ---
            if metrics_buffer["f1"]:
                # 统一放大 100 倍，以百分比展示
                m_f1, s_f1 = np.mean(metrics_buffer["f1"])*100, np.std(metrics_buffer["f1"])*100
                m_acc, s_acc = np.mean(metrics_buffer["acc"])*100, np.std(metrics_buffer["acc"])*100
                m_prec, s_prec = np.mean(metrics_buffer["prec"])*100, np.std(metrics_buffer["prec"])*100
                m_rec, s_rec = np.mean(metrics_buffer["rec"])*100, np.std(metrics_buffer["rec"])*100
                
                print(f"Result: Acc={m_acc:.1f}±{s_acc:.1f} | P={m_prec:.1f}±{s_prec:.1f} | R={m_rec:.1f}±{s_rec:.1f} | F1={m_f1:.1f}±{s_f1:.1f}")
                
                final_summary.append({
                    "granularity": granularity,
                    "features": combo_key,
                    "metrics": {
                        "accuracy": f"{m_acc:.2f}±{s_acc:.2f}",
                        "precision": f"{m_prec:.2f}±{s_prec:.2f}",
                        "recall": f"{m_rec:.2f}±{s_rec:.2f}",
                        "f1": f"{m_f1:.2f}±{s_f1:.2f}",
                        # 保留原始浮点数方便后续画图或进一步处理
                        "raw_mean": {"acc": m_acc, "prec": m_prec, "rec": m_rec, "f1": m_f1},
                        "raw_std": {"acc": s_acc, "prec": s_prec, "rec": s_rec, "f1": s_f1}
                    }
                })

    # ================= 6. 保存总结与顶会风格输出 =================
    # 按 F1 的均值降序排列
    final_summary.sort(key=lambda x: x['metrics']['raw_mean']['f1'], reverse=True)
    
    print("\n\n" + "="*110)
    print(f"{'FINAL ABLATION LEADERBOARD':^110}")
    print("="*110)
    
    # 完美的排版表头
    header = f"{'Mode':<12} | {'Features':<35} | {'Acc (Mean±Std)':<14} | {'Prec (Mean±Std)':<14} | {'Rec (Mean±Std)':<14} | {'F1 (Mean±Std)':<14}"
    print(header)
    print("-" * 110)
    
    for item in final_summary:
        mode = item['granularity']
        feats = item['features']
        # 如果特征名字太长，截断一下保证表格对齐
        if len(feats) > 33:
            feats = feats[:30] + "..."
            
        acc_str = item['metrics']['accuracy']
        prec_str = item['metrics']['precision']
        rec_str = item['metrics']['recall']
        f1_str = item['metrics']['f1']
        
        row = f"{mode:<12} | {feats:<35} | {acc_str:<14} | {prec_str:<14} | {rec_str:<14} | {f1_str:<14}"
        print(row)
    
    # 剔除冗余的 raw_mean 和 raw_std 再保存，让 JSON 更干净
    for item in final_summary:
        del item['metrics']['raw_mean']
        del item['metrics']['raw_std']
        
    with open(SUMMARY_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_summary, f, indent=4, ensure_ascii=False)

    print("="*110)
    print(f"All done! Detailed summary saved to: {SUMMARY_FILE}")