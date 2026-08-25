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
MODEL_PATH = "/root/autodl-tmp/GLM-4-32B-Chat" # 替换为你的实际路径
DATA_FILE = "/root/autodl-tmp/DATA/updated_test.jsonl"
BASE_OUTPUT_DIR = "/root/autodl-tmp/non-fine-tuning/ChatGLM/multi_ablation"
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

def map_to_binary(score):
    """将 0-5 的多分类分数映射为 2分类 (0-1 -> 0负类, 2-5 -> 1正类)"""
    return 0 if score <= 1 else 1

def parse_model_output(response):
    """解析模型输出，提取 0-5 的数字"""
    response = response.strip()
    nums = re.findall(r'\d+', response)
    if nums:
        val = int(nums[0])
        return max(0, min(5, val)) # 限制在 0-5 之间
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

    # --- D. 组装最终 Prompt ---
    prompt = f"""### Role
You are an expert linguist specializing in pragmatics and moral coercion detection.

### Task
Analyze the dialogue to determine the severity of moral coercion.
Use the provided linguistic feature signals as expert hints.

### Feature Definitions
{def_block}

### Input Data
{global_block}
Dialogue Transcript:
{dialogue_text}

### Output Requirement
Assess the severity of moral coercion on a scale from 0 to 5.
0: No moral coercion
1: Slight suspicion (Ambiguous)
2: Slight tendency (Weak)
3: Clear implication (Moderate)
4: Highly coercive (Strong)
5: Extreme moral coercion (Severe)

Your Answer (output only a single number between 0 and 5):
"""
    return prompt

if __name__ == "__main__":
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    
    print(f"Loading GLM model with vLLM from: {MODEL_PATH}")
    # ================= 4. vLLM 引擎初始化 =================
    llm = LLM(
        model=MODEL_PATH, 
        tensor_parallel_size=4,      # 4 张 4090 齐上阵，完美整除 65024！
        dtype="bfloat16",            # 恢复满血半精度，告别量化损耗
        trust_remote_code=True,      
        gpu_memory_utilization=0.90, # 96GB 跑 64GB 模型很充裕，留 10% 即可
        max_model_len=2048,          
        enforce_eager=True           
    )
    
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
            
            metrics_multi = {"acc": [], "f1": [], "prec": [], "rec": []}
            metrics_bin   = {"acc": [], "f1": [], "prec": [], "rec": []}
            
            for seed in SEEDS:
                # 在采样参数中设置温度和种子
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
                true_multi_labels = []
                
                for item in all_data:
                    prompt_text = generate_prompt_content(item, combo, granularity)
                    
                    messages = [
                        {"role": "system", "content": "You are an expert linguist specializing in pragmatics."},
                        {"role": "user", "content": prompt_text}
                    ]
                    
                    # 使用 tokenizer 应用 GLM-4 的对话模板
                    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    
                    prompts.append(text)
                    dialogue_ids.append(item.get("dialogue_id"))
                    # 兼容不同命名格式的真实标签
                    true_multi_labels.append(item.get("label_m", item.get("dialog_multi_label", -1)))
                
                # --- vLLM 高并发推理 ---
                print(f"  [Seed {seed}] Inferencing {len(prompts)} samples...")
                outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
                
                # --- 结果解析 ---
                results = []
                for i, output in enumerate(outputs):
                    response = output.outputs[0].text.strip()
                    pred_multi = parse_model_output(response)
                    
                    results.append({
                        "id": dialogue_ids[i],
                        "true": true_multi_labels[i],
                        "pred": pred_multi,
                        "raw": response
                    })
                
                # 保存单次推理结果
                with open(output_file, 'w', encoding='utf-8') as f:
                    for r in results:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
                
                # --- 指标计算 ---
                valid_res = [r for r in results if r["pred"] != -1 and r["true"] != -1]
                if valid_res:
                    # 1. 提取 6分类标签计算 Macro 指标
                    y_true_multi = [r["true"] for r in valid_res]
                    y_pred_multi = [r["pred"] for r in valid_res]
                    
                    acc_m = accuracy_score(y_true_multi, y_pred_multi)
                    p_m, r_m, f1_m, _ = precision_recall_fscore_support(
                        y_true_multi, y_pred_multi, average='macro', zero_division=0
                    )
                    metrics_multi["acc"].append(acc_m)
                    metrics_multi["f1"].append(f1_m)
                    metrics_multi["prec"].append(p_m)
                    metrics_multi["rec"].append(r_m)
                    
                    # 2. 映射成 2分类标签计算 Binary 指标
                    y_true_bin = [map_to_binary(y) for y in y_true_multi]
                    y_pred_bin = [map_to_binary(y) for y in y_pred_multi]
                    
                    acc_b = accuracy_score(y_true_bin, y_pred_bin)
                    p_b, r_b, f1_b, _ = precision_recall_fscore_support(
                        y_true_bin, y_pred_bin, average='binary', zero_division=0
                    )
                    metrics_bin["acc"].append(acc_b)
                    metrics_bin["f1"].append(f1_b)
                    metrics_bin["prec"].append(p_b)
                    metrics_bin["rec"].append(r_b)

            # --- 当前组合 3 个 Seed 汇总 ---
            if metrics_bin["f1"]:
                print(f"  -> Result (Binary): Acc={np.mean(metrics_bin['acc']):.4f} | Prec={np.mean(metrics_bin['prec']):.4f} | Rec={np.mean(metrics_bin['rec']):.4f} | F1={np.mean(metrics_bin['f1']):.4f}")
                
                final_summary.append({
                    "granularity": granularity,
                    "features": combo_key,
                    "metrics_binary": {
                        "acc_mean": np.mean(metrics_bin["acc"]), "acc_std": np.std(metrics_bin["acc"]),
                        "prec_mean": np.mean(metrics_bin["prec"]), "prec_std": np.std(metrics_bin["prec"]),
                        "rec_mean": np.mean(metrics_bin["rec"]), "rec_std": np.std(metrics_bin["rec"]),
                        "f1_mean": np.mean(metrics_bin["f1"]), "f1_std": np.std(metrics_bin["f1"])
                    },
                    "metrics_multi": {
                        "acc_mean": np.mean(metrics_multi["acc"]), "acc_std": np.std(metrics_multi["acc"]),
                        "prec_mean": np.mean(metrics_multi["prec"]), "prec_std": np.std(metrics_multi["prec"]),
                        "rec_mean": np.mean(metrics_multi["rec"]), "rec_std": np.std(metrics_multi["rec"]),
                        "f1_mean": np.mean(metrics_multi["f1"]), "f1_std": np.std(metrics_multi["f1"])
                    },
                    "sort_score": np.mean(metrics_bin["f1"])
                })

    # ================= 6. 保存总结 =================
    final_summary.sort(key=lambda x: x['sort_score'], reverse=True)
    
    print("\n" + "="*115)
    print("FINAL ABLATION LEADERBOARD (Sorted by Binary F1)")
    print("="*115)
    print(f"{'Mode':<12} | {'Features':<32} | {'B-Acc':<6} | {'B-Pre':<6} | {'B-Rec':<6} | {'B-F1':<6} | {'M-Acc':<6} | {'M-Pre':<6} | {'M-Rec':<6} | {'M-F1':<6}")
    print("-" * 115)
    for item in final_summary:
        b = item['metrics_binary']
        m = item['metrics_multi']
        mode = item['granularity']
        feat = item['features']
        print(f"{mode:<12} | {feat:<32} | {b['acc_mean']:.4f} | {b['prec_mean']:.4f} | {b['rec_mean']:.4f} | {b['f1_mean']:.4f} | {m['acc_mean']:.4f} | {m['prec_mean']:.4f} | {m['rec_mean']:.4f} | {m['f1_mean']:.4f}")
    
    with open(SUMMARY_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_summary, f, indent=4, ensure_ascii=False)

    print(f"\nAll done! Summary saved to {SUMMARY_FILE}")