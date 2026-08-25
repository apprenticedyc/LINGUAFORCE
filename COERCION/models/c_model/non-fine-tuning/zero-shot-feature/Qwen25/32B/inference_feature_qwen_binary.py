import json
import re
import itertools
import os
import random
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from vllm import LLM, SamplingParams

# ================= 1. 配置区域 (Configuration) =================
# 模型路径 (确保与您服务器上的实际文件夹名一致)
MODEL_PATH = "/root/autodl-tmp/Qwen25-32B-Instruct" 
# 数据路径
DATA_FILE = "/root/autodl-tmp/DATA/updated_test.jsonl"
# 结果保存路径
BASE_OUTPUT_DIR = "/root/autodl-tmp/non-fine-tuning/Qwen25_32B_feature/binary_ablation"
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

def parse_model_output(response):
    response = response.strip()
    if response in ["0", "1"]:
        return int(response)
    nums = re.findall(r'\b[01]\b', response)
    if nums:
        return int(nums[0])
    return -1

def get_feature_display_str(val_dict):
    """仅处理结构化的 dict 数据"""
    if not isinstance(val_dict, dict):
        return None
        
    level = val_dict.get("level", "None")
    score = val_dict.get("intensity_score", 0.0)
    
    # 过滤掉无意义的特征
    if level != "None" or score > 0.05:
        return f"{level} (Intensity: {score:.2f})"
    return None

# ================= 3. 核心逻辑：动态 Prompt 生成 =================

def generate_prompt_content(item, active_indices, granularity):
    """基于 active_indices 和 granularity 动态生成 Prompt"""
    
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
    
    print(f"Loading model: {MODEL_PATH} with vLLM across 3 GPUs...")
    # ================= vLLM 引擎初始化 =================
    llm = LLM(
        model=MODEL_PATH, 
        tensor_parallel_size=1,      # 不切分注意力头
        pipeline_parallel_size=3,    # 纵向层切分，分配到 3 张 5090
        dtype="bfloat16",         
        trust_remote_code=True,
        gpu_memory_utilization=0.9,
        max_model_len=8192,          # 限制显存预分配
        enforce_eager=True           # 禁用 CUDA 图，防止多卡死锁
    )
    
    # 提取模型自带的 tokenizer，用于套用 Chat Template
    tokenizer = llm.get_tokenizer()

    print(f"Loading data: {DATA_FILE}")
    all_data = []
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                all_data.append(json.loads(line.strip()))
    
    if MAX_SAMPLES:
        all_data = all_data[:MAX_SAMPLES]
        print(f"Debug: Truncated to {MAX_SAMPLES} samples.")

    # 生成特征组合
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
            
            print(f"\n--- Combo: {combo_key} ---")
            metrics_buffer = {"acc": [], "f1": [], "prec": [], "rec": []}
            
            for seed in SEEDS:
                # vLLM 需要在 SamplingParams 中设置 seed 来保证生成过程的确定性
                sampling_params = SamplingParams(
                    temperature=0.7,
                    top_p=0.9,
                    max_tokens=5,
                    seed=seed 
                )
                
                seed_dir = os.path.join(BASE_OUTPUT_DIR, f"seed_{seed}")
                os.makedirs(seed_dir, exist_ok=True)
                output_file = os.path.join(seed_dir, f"res_{granularity}_{safe_filename_key}.jsonl")
                
                # --- 构建批处理 Prompt 列表 ---
                prompts = []
                dialogue_ids = []
                true_labels = []
                
                for item in all_data:
                    prompt_text = generate_prompt_content(item, combo, granularity)
                    messages = [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt_text}
                    ]
                    # 使用 Qwen 官方的 Chat 模板
                    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    prompts.append(text)
                    dialogue_ids.append(item.get("dialogue_id"))
                    true_labels.append(item.get("dialog_binary_label", -1))
                
                # --- vLLM 高并发推理 ---
                print(f"  [Seed {seed}] Inferencing {len(prompts)} samples...")
                outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
                
                # --- 结果解析 ---
                results = []
                for i, output in enumerate(outputs):
                    response = output.outputs[0].text
                    pred = parse_model_output(response)
                    
                    results.append({
                        "id": dialogue_ids[i],
                        "true": true_labels[i],
                        "pred": pred,
                        "raw": response.strip()
                    })
                
                # 保存单次推理结果
                with open(output_file, 'w', encoding='utf-8') as f:
                    for r in results:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
                
                # 计算指标
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
            
            # --- 计算并打印当前组合的统计信息 ---
            if metrics_buffer["f1"]:
                mean_f1 = np.mean(metrics_buffer["f1"])
                std_f1 = np.std(metrics_buffer["f1"])
                mean_acc = np.mean(metrics_buffer["acc"])
                std_acc = np.std(metrics_buffer["acc"])
                mean_prec = np.mean(metrics_buffer["prec"])
                std_prec = np.std(metrics_buffer["prec"])
                mean_rec = np.mean(metrics_buffer["rec"])
                std_rec = np.std(metrics_buffer["rec"])
                
                print(f"  -> Result: F1 = {mean_f1:.4f} ± {std_f1:.4f} | Acc = {mean_acc:.4f} ± {std_acc:.4f}")
                
                final_summary.append({
                    "granularity": granularity,
                    "features": combo_key,
                    "metrics": {
                        "f1_mean": mean_f1, "f1_std": std_f1,
                        "acc_mean": mean_acc, "acc_std": std_acc,
                        "prec_mean": mean_prec, "prec_std": std_prec,
                        "rec_mean": mean_rec, "rec_std": std_rec
                    }
                })

    # ================= 6. 保存总结 =================
    final_summary.sort(key=lambda x: x['metrics']['f1_mean'], reverse=True)
    
    print("\n" + "="*115)
    print("FINAL ABLATION LEADERBOARD (Mean ± Std)")
    print("="*115)
    header = f"{'Mode':<12} | {'Features':<40} | {'Accuracy':<18} | {'Precision':<18} | {'Recall':<18} | {'F1 Score':<18}"
    print(header)
    print("-" * 115)
    
    for item in final_summary:
        m = item['metrics']
        acc_str = f"{m['acc_mean']:.4f} ± {m['acc_std']:.4f}"
        prec_str = f"{m['prec_mean']:.4f} ± {m['prec_std']:.4f}"
        rec_str = f"{m['rec_mean']:.4f} ± {m['rec_std']:.4f}"
        f1_str = f"{m['f1_mean']:.4f} ± {m['f1_std']:.4f}"
        
        print(f"{item['granularity']:<12} | {item['features']:<40} | {acc_str:<18} | {prec_str:<18} | {rec_str:<18} | {f1_str:<18}")
    
    with open(SUMMARY_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_summary, f, indent=4, ensure_ascii=False)

    print(f"\nAll done! Summary saved to {SUMMARY_FILE}")