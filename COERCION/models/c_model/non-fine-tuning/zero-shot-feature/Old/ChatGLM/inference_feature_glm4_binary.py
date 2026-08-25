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

# ================= 配置区域 =================
# 【修改点 1】模型路径改为 GLM-4 (请确认您的实际路径，通常是 glm-4-9b-chat)
model_name = "/root/autodl-tmp/glm-4-9b-chat" 

data_file = "/root/autodl-tmp/DATA/test.jsonl"

# 【修改点 2】输出目录改为 ChatGLM
base_output_dir = "/root/autodl-tmp/non-fine-tuning/ChatGLM/feature_results_binary"
summary_file = os.path.join(base_output_dir, "final_metrics_summary_stats.json")

# 设定种子列表
SEEDS = [42, 519, 2026]

# 创建输出目录
os.makedirs(base_output_dir, exist_ok=True)

# 限制测试样本数 (None 表示跑全量，调试时可设为 10)
max_samples = None 

feature_map = {
    0: ("Obligation", "Likelihood of expressing duty/necessity"),
    1: ("Constraint", "Likelihood of restricting the listener's choices"),
    2: ("Value Judgement", "Likelihood of making moral evaluations"),
    3: ("Toxicity", "Likelihood of offensive/hateful language")
}

# 设置随机种子的工具函数
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
# ===========================================

# 1. 加载模型
print(f"Loading GLM-4 model from {model_name}...")

# 【修改点 3】GLM-4 必须加 trust_remote_code=True
tokenizer = AutoTokenizer.from_pretrained(
    model_name, 
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True  # 【重要】必须加
)

# 2. 读取数据
print(f"Loading data from {data_file}...")
all_data = []
if os.path.exists(data_file):
    with open(data_file, 'r', encoding='utf-8') as f:
        for line in f:
            all_data.append(json.loads(line.strip()))
else:
    print(f"Error: Data file not found at {data_file}")
    all_data = []

if max_samples:
    all_data = all_data[:max_samples]
    print(f"Debug mode: Testing on first {max_samples} samples.")

# 3. 生成所有特征组合
feature_indices = [0, 1, 2, 3]
combinations = []
for r in range(1, 5): 
    combinations.extend(itertools.combinations(feature_indices, r))

print(f"Total combinations: {len(combinations)} | Seeds per combo: {len(SEEDS)}")

# 最终汇总列表
final_stats_summary = []

# 4. 主循环：外层遍历组合，内层遍历种子
for combo in combinations:
    combo_names = [feature_map[idx][0] for idx in combo]
    combo_key = "+".join(combo_names)
    
    print(f"\n{'='*60}")
    print(f"Experiment: {combo_key}")
    print(f"{'='*60}")
    
    # 临时存储当前组合下 3 个种子的指标
    current_combo_metrics = {
        "acc": [], "prec": [], "rec": [], "f1": []
    }
    
    for seed in SEEDS:
        # 设置种子
        set_seed(seed)
        
        # 为每个种子/组合创建独立的文件路径
        seed_dir = os.path.join(base_output_dir, f"seed_{seed}")
        os.makedirs(seed_dir, exist_ok=True)
        output_filename = os.path.join(seed_dir, f"result_{combo_key}.jsonl")
        
        results = []
        
        # --- 推理过程 ---
        for item in tqdm(all_data, desc=f"Seed {seed} | {combo_key}", leave=False):
            dialogue_id = item["dialogue_id"]
            utterances = item["utterances"]
            true_binary = item.get("dialog_binary_label", -1)
            d_feats = item.get("dialog_features", [0.0, 0.0, 0.0, 0.0])
            
            # 构建 Feature Block
            feature_context_parts = []
            for idx in combo:
                name, desc = feature_map[idx]
                val = d_feats[idx]
                feature_context_parts.append(f"- {name}: {val:.4f} ({desc})")
            
            feature_block = "\n".join(feature_context_parts)
            dialogue_text = "\n".join([f"Turn {j+1}: {utt}" for j, utt in enumerate(utterances) if utt.strip()])
            
            prompt_content = f"""
Moral coercion is a subtle form of pressure where the speaker uses moral obligation, guilt, or social norms to force compliance.

Input Data:
1. Linguistic Feature Signals (Probabilities 0.0 - 1.0):
{feature_block}

2. Dialogue Transcript:
{dialogue_text}

Task:
Based on the dialogue content AND the provided linguistic signals, determine if this is a case of moral coercion.
Your Answer 1 if YES, 0 if NO (Number only).
"""
            messages = [
                {"role": "system", "content": "You are an expert linguist specializing in pragmatics."},
                {"role": "user", "content": prompt_content}
            ]
            
            # 【修改点 4】GLM-4 推荐的推理写法
            # 直接使用 tokenize=True 返回 tensor，避免手动处理
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
                    **inputs,  # 直接解包传入 input_ids 和 attention_mask
                    max_new_tokens=5,
                    do_sample=True,       
                    temperature=0.7,      
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id 
                )
            
            # 解码
            output_ids = generated_ids[:, input_len:]
            response = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
            
            # 结果解析逻辑 (保持不变，很健壮)
            pred_binary = -1
            if response in ["0", "1"]:
                pred_binary = int(response)
            else:
                nums = re.findall(r'\b[01]\b', response)
                if nums:
                    pred_binary = int(nums[0])
            
            results.append({
                "dialogue_id": dialogue_id,
                "true_binary": true_binary,
                "pred_binary": pred_binary,
                "raw_response": response
            })
            
        # --- 单个种子计算指标 ---
        valid_results = [r for r in results if r["pred_binary"] != -1]
        
        acc, precision, recall, f1 = 0.0, 0.0, 0.0, 0.0
        
        if len(valid_results) > 0:
            y_true = [r["true_binary"] for r in valid_results]
            y_pred = [r["pred_binary"] for r in valid_results]
            
            acc = accuracy_score(y_true, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average='binary', zero_division=0
            )
            
            # 保存该种子结果
            with open(output_filename, 'w', encoding='utf-8') as f:
                for r in results:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        
        current_combo_metrics["acc"].append(acc)
        current_combo_metrics["prec"].append(precision)
        current_combo_metrics["rec"].append(recall)
        current_combo_metrics["f1"].append(f1)
        
        print(f"  -> Seed {seed:<4} | Acc: {acc:.4f} | Prec: {precision:.4f} | Rec: {recall:.4f} | F1: {f1:.4f}")

        # 【建议】手动清理显存，GLM-4 跑长循环时偶尔会碎片化
        torch.cuda.empty_cache()

    # --- 所有种子跑完，计算统计值 ---
    mean_acc = np.mean(current_combo_metrics["acc"])
    std_acc = np.std(current_combo_metrics["acc"])
    
    mean_prec = np.mean(current_combo_metrics["prec"])
    std_prec = np.std(current_combo_metrics["prec"])
    
    mean_rec = np.mean(current_combo_metrics["rec"])
    std_rec = np.std(current_combo_metrics["rec"])
    
    mean_f1 = np.mean(current_combo_metrics["f1"])
    std_f1 = np.std(current_combo_metrics["f1"])

    str_acc = f"{mean_acc:.4f}±{std_acc:.4f}"
    str_prec = f"{mean_prec:.4f}±{std_prec:.4f}"
    str_rec = f"{mean_rec:.4f}±{std_rec:.4f}"
    str_f1 = f"{mean_f1:.4f}±{std_f1:.4f}"

    print("-" * 100)
    header = f"{'Metric Analysis':<25} | {'Acc (Mean±Std)':<18} | {'Prec (Mean±Std)':<18} | {'Rec (Mean±Std)':<18} | {'F1 (Mean±Std)':<18}"
    values = f"{'Results':<25} | {str_acc:<18} | {str_prec:<18} | {str_rec:<18} | {str_f1:<18}"
    
    print(header)
    print(values)
    print("-" * 100)

    final_stats_summary.append({
        "features": combo_key,
        "metrics_mean": {
            "accuracy": mean_acc, "precision": mean_prec, "recall": mean_rec, "f1": mean_f1
        },
        "metrics_std": {
            "accuracy": std_acc, "precision": std_prec, "recall": std_rec, "f1": std_f1
        },
        "metrics_formatted": {
            "accuracy": str_acc, "precision": str_prec, "recall": str_rec, "f1": str_f1
        }
    })

# 5. 保存并打印最终排行榜
print("\n" + "="*40)
print("Final Summary (Sorted by Mean F1)")
print("="*40)

final_stats_summary.sort(key=lambda x: x['metrics_mean']['f1'], reverse=True)

print(f"{'Features':<40} | {'F1 (Mean±Std)':<20} | {'Acc (Mean±Std)':<20}")
print("-" * 85)
for item in final_stats_summary:
    f_name = item['features']
    f1_str = item['metrics_formatted']['f1']
    acc_str = item['metrics_formatted']['accuracy']
    print(f"{f_name:<40} | {f1_str:<20} | {acc_str:<20}")

with open(summary_file, 'w', encoding='utf-8') as f:
    json.dump(final_stats_summary, f, indent=4, ensure_ascii=False)

print(f"\nDetailed statistics saved to {summary_file}")