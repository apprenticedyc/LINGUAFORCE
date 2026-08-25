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
# 模型路径
MODEL_PATH = "/root/autodl-tmp/Qwen25-7B-Instruct" 
# 数据路径
DATA_FILE = "/root/autodl-tmp/DATA/updated_test.jsonl"
# 结果保存路径
BASE_OUTPUT_DIR = "/root/autodl-tmp/non-fine-tuning/Qwen25/binary_asymmetric"
SUMMARY_FILE = os.path.join(BASE_OUTPUT_DIR, "final_asymmetric_summary.json")

# 随机种子
SEEDS = [42, 520, 2026]

# 最大测试样本数 (警告：255种组合极其耗时，建议先设为 100 测试！)
MAX_SAMPLES = 100 # TODO: 正式跑实验时请改为 None

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
    """提取结构化特征分数"""
    if not isinstance(val_dict, dict):
        return None
    level = val_dict.get("level", "None")
    score = val_dict.get("intensity_score", 0.0)
    if level != "None" or score > 0.05:
        return f"{level} (Intensity: {score:.2f})"
    return None

def get_config_name(config):
    """ 将状态元组 (e.g., (1,3,0,2)) 转换为易读的组合名称 G:[OB+CS] | T:[CS+TX] """
    global_feats = [FEATURE_NAMES[i] for i, state in enumerate(config) if state in (1, 3)]
    turn_feats = [FEATURE_NAMES[i] for i, state in enumerate(config) if state in (2, 3)]
    
    g_str = "G:[" + "+".join(global_feats) + "]" if global_feats else "G:[None]"
    t_str = "T:[" + "+".join(turn_feats) + "]" if turn_feats else "T:[None]"
    return f"{g_str} | {t_str}"

# ================= 3. 核心逻辑：非对称动态 Prompt 生成 =================

def generate_prompt_content(item, config):
    """
    基于非对称路由配置生成 Prompt
    config: 一个长度为4的元组，代表4个特征的状态。0=None, 1=Global, 2=Turn, 3=Both
    """
    global_active = [i for i, state in enumerate(config) if state in (1, 3)]
    turn_active = [i for i, state in enumerate(config) if state in (2, 3)]
    all_active = list(set(global_active + turn_active))

    # --- A. 生成定义部分 ---
    definitions = []
    for idx in all_active:
        name, desc = FEATURE_DEFINITIONS[idx]
        definitions.append(f"- {name}: {desc}")
    def_block = "\n".join(definitions) if definitions else "None"

    # --- B. 生成 Global 特征块 ---
    global_block = ""
    if global_active:
        lines = []
        overall = item.get("overall_analysis", {})
        for idx in global_active:
            name_key = FEATURE_NAMES[idx]
            if name_key in overall:
                feat_str = get_feature_display_str(overall[name_key])
                if feat_str:
                    lines.append(f"- {name_key}: {feat_str}")
        if lines:
            global_block = "Global Analysis Signals:\n" + "\n".join(lines) + "\n"
        # else:
        #     global_block = "Global Analysis Signals: None detected.\n"
    # else:
    #     global_block = "Global Analysis Signals: None provided.\n"

    # --- C. 生成 Turn-level 特征块 ---
    utterances = item.get("utterances", [])
    dialogue_lines = []
    turn_analysis = item.get("turn_analysis", [])
    structured_turn_map = {t['turn_id']: t.get('features', {}) for t in turn_analysis}

    for i, text in enumerate(utterances):
        speaker = "Person1" if i % 2 == 0 else "Person2"
        line = f"Turn {i} ({speaker}): \"{text}\""
        if turn_active:
            turn_signals = []
            for idx in turn_active:
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

# ================= 4. 主程序执行 =================

if __name__ == "__main__":
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    
    print(f"Loading model: {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, 
        torch_dtype=torch.bfloat16, 
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
        print(f"Debug Mode: Truncated to {MAX_SAMPLES} samples to save time.")

    # --- 生成 255 种非对称特征组合 ---
    # 状态码: 0=None, 1=Global, 2=Turn, 3=Both
    states = [0, 1, 2, 3]
    all_configs = list(itertools.product(states, repeat=4))
    all_configs = [c for c in all_configs if c != (0, 0, 0, 0)] # 排除纯 Zero-shot
    
    print(f"Total Asymmetric Combinations to run: {len(all_configs)}")
    
    final_summary = []

    # ================= 5. 单层遍历 255 个实验 =================
    
    for config in all_configs:
        combo_name = get_config_name(config)
        safe_filename_key = combo_name.replace(":", "").replace("[", "").replace("]", "").replace("+", "_").replace(" ", "").replace("|", "AND")
        
        print(f"\n{'#'*80}")
        print(f"Running Config: {combo_name}")
        print(f"{'#'*80}")
        
        metrics_buffer = {"acc": [], "f1": [], "prec": [], "rec": []}
        
        for seed in SEEDS:
            set_seed(seed)
            seed_dir = os.path.join(BASE_OUTPUT_DIR, f"seed_{seed}")
            os.makedirs(seed_dir, exist_ok=True)
            output_file = os.path.join(seed_dir, f"res_asym_{safe_filename_key}.jsonl")
            
            results = []
            
            for item in tqdm(all_data, desc=f"Seed {seed}", leave=False):
                prompt_text = generate_prompt_content(item, config)
                print(prompt_text)
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt_text}
                ]
                
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
                
                with torch.no_grad():
                    generated_ids = model.generate(
                        model_inputs.input_ids,
                        attention_mask=model_inputs.attention_mask,
                        max_new_tokens=5,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        pad_token_id=tokenizer.eos_token_id
                    )
                
                output_ids = generated_ids[:, model_inputs.input_ids.shape[1]:]
                response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
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
                
        if metrics_buffer["f1"]:
            mean_f1 = np.mean(metrics_buffer["f1"])
            std_f1 = np.std(metrics_buffer["f1"])
            mean_acc = np.mean(metrics_buffer["acc"])
            std_acc = np.std(metrics_buffer["acc"])
            mean_prec = np.mean(metrics_buffer["prec"])
            std_prec = np.std(metrics_buffer["prec"])
            mean_rec = np.mean(metrics_buffer["rec"])
            std_rec = np.std(metrics_buffer["rec"])
            
            print(f"Result: F1 = {mean_f1:.4f} ± {std_f1:.4f} | Acc = {mean_acc:.4f} ± {std_acc:.4f}")
            
            final_summary.append({
                "config_name": combo_name, 
                "config_tuple": config,
                "metrics": {
                    "f1_mean": mean_f1, "f1_std": std_f1,
                    "acc_mean": mean_acc, "acc_std": std_acc,
                    "prec_mean": mean_prec, "prec_std": std_prec,
                    "rec_mean": mean_rec, "rec_std": std_rec
                }
            })

    # ================= 6. 保存总结与输出榜单 =================
    final_summary.sort(key=lambda x: x['metrics']['f1_mean'], reverse=True)
    
    print("\n" + "="*130)
    print("FINAL ASYMMETRIC ABLATION LEADERBOARD (Mean ± Std)")
    print("="*130)
    header = f"{'Configuration (Global | Turn)':<60} | {'Accuracy':<15} | {'Precision':<15} | {'Recall':<15} | {'F1 Score':<15}"
    print(header)
    print("-" * 130)
    
    for item in final_summary:
        m = item['metrics']
        acc_str = f"{m['acc_mean']:.4f} ± {m['acc_std']:.4f}"
        prec_str = f"{m['prec_mean']:.4f} ± {m['prec_std']:.4f}"
        rec_str = f"{m['rec_mean']:.4f} ± {m['rec_std']:.4f}"
        f1_str = f"{m['f1_mean']:.4f} ± {m['f1_std']:.4f}"
        
        print(f"{item['config_name']:<60} | {acc_str:<15} | {prec_str:<15} | {rec_str:<15} | {f1_str:<15}")
    
    with open(SUMMARY_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_summary, f, indent=4, ensure_ascii=False)

    print(f"\nAll 255 combinations evaluated! Summary saved to {SUMMARY_FILE}")