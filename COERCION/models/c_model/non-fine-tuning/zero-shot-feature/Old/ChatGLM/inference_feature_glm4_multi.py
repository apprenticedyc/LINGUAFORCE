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
# [修改 1] 模型路径改为 GLM-4
model_name = "/root/autodl-tmp/glm-4-9b-chat" 

data_file = "/root/autodl-tmp/DATA/test.jsonl"

# [修改 2] 输出目录名改为 ChatGLM
base_output_dir = "/root/autodl-tmp/non-fine-tuning/ChatGLM/feature_results_multi" 
summary_file = os.path.join(base_output_dir, "final_multi_summary.json")

# 设定种子列表
SEEDS = [42, 519, 2026]

# 创建输出目录
os.makedirs(base_output_dir, exist_ok=True)

# 限制测试样本数 (None 为跑全量)
max_samples = None 

feature_map = {
    0: ("Obligation", "Likelihood of expressing duty/necessity"),
    1: ("Constraint", "Likelihood of restricting the listener's choices"),
    2: ("Value Judgement", "Likelihood of making moral evaluations"),
    3: ("Toxicity", "Likelihood of offensive/hateful language")
}

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# 映射函数：0-5 -> 0/1
def map_to_binary(score):
    return 0 if score <= 1 else 1

# ===========================================

# 1. 加载模型
print(f"Loading model from {model_name}...")

# [修改 3] 加载 Tokenizer，必须加 trust_remote_code=True
tokenizer = AutoTokenizer.from_pretrained(
    model_name, 
    trust_remote_code=True
)

# [修改 4] 加载 Model，必须加 trust_remote_code=True
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto"
).eval() # 显式设为 eval 模式

# 2. 读取数据
print(f"Loading data from {data_file}...")
all_data = []
with open(data_file, 'r', encoding='utf-8') as f:
    for line in f:
        try:
            all_data.append(json.loads(line.strip()))
        except:
            continue

if max_samples:
    all_data = all_data[:max_samples]
    print(f"Debug mode: Testing on first {max_samples} samples.")

# 3. 生成特征组合
feature_indices = [0, 1, 2, 3]
combinations = []
for r in range(1, 5): 
    combinations.extend(itertools.combinations(feature_indices, r))

print(f"Total combinations: {len(combinations)}")

final_stats_summary = []

# 4. 主循环
for combo in combinations:
    combo_names = [feature_map[idx][0] for idx in combo]
    combo_key = "+".join(combo_names)
    
    print(f"\n{'='*80}")
    print(f"Experiment: {combo_key}")
    print(f"{'='*80}")
    
    # 初始化两套指标的容器
    metrics_multi = {"acc": [], "prec": [], "rec": [], "f1": []}
    metrics_bin   = {"acc": [], "prec": [], "rec": [], "f1": []}
    
    for seed in SEEDS:
        set_seed(seed)
        
        # 结果保存路径
        seed_dir = os.path.join(base_output_dir, f"seed_{seed}")
        os.makedirs(seed_dir, exist_ok=True)
        output_filename = os.path.join(seed_dir, f"result_{combo_key}.jsonl")
        
        results = []
        y_true_multi_list = []
        y_pred_multi_list = []
        
        # --- 推理过程 ---
        for item in tqdm(all_data, desc=f"Seed {seed} | {combo_key}", leave=False):
            dialogue_id = item["dialogue_id"]
            utterances = item.get("utts", item.get("utterances", []))
            
            # 获取真实 6分类 标签
            true_multi = item.get("label_m", item.get("dialog_multi_label", 0))
            d_feats = item.get("dialog_features", [0.0]*4)
            
            # 构建 Feature Block
            feature_context_parts = []
            for idx in combo:
                name, desc = feature_map[idx]
                val = d_feats[idx]
                feature_context_parts.append(f"- {name}: {val:.4f} ({desc})")
            feature_block = "\n".join(feature_context_parts)
            
            dialogue_text = "\n".join([f"Turn {j+1}: {utt}" for j, utt in enumerate(utterances) if utt.strip()])
            
            # Prompt: 要求 0-5
            prompt_content = f"""
Moral coercion is a subtle form of pressure where the speaker uses moral obligation, guilt, or social norms to force compliance.

Input Data:
1. Linguistic Feature Signals (Probabilities 0.0 - 1.0):
{feature_block}

2. Dialogue Transcript:
{dialogue_text}

Task:
Based on the dialogue content AND the provided linguistic signals, assess the severity of moral coercion on a scale from 0 to 5.
(0 indicates no moral coercion, 5 indicates extreme moral coercion)
Severity Levels:
0: No moral coercion
1: Slight suspicion
2: Slight tendency
3: Clear implication
4: Highly self-centered / Coercive
5: Extreme moral coercion

Your Answer (output only a single number between 0 and 5):
"""
            # GLM-4 也支持这种标准的 system/user 格式
            messages = [
                {"role": "system", "content": "You are an expert linguist specializing in pragmatics."},
                {"role": "user", "content": prompt_content}
            ]
            
            # [修改 5] 确保 apply_chat_template 适用于 GLM-4
            # GLM-4 的 tokenizer 通常会自动处理 special tokens，但 return_dict=True 和 add_generation_prompt=True 是比较稳妥的写法
            inputs = tokenizer.apply_chat_template(
                messages, 
                add_generation_prompt=True, 
                tokenize=True, 
                return_tensors="pt",
                return_dict=True
            ).to(model.device)
            
            # 生成参数
            gen_kwargs = {
                "max_new_tokens": 5,
                "do_sample": True,
                "temperature": 0.7,
                "top_p": 0.9,
                "pad_token_id": tokenizer.eos_token_id # 防止 pad 警告
            }

            with torch.no_grad():
                outputs = model.generate(**inputs, **gen_kwargs)
            
            # [修改 6] 解码 (GLM-4 输出包含输入部分，需要切片)
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
            response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            
            # 解析 0-5
            nums = re.findall(r'\d+', response)
            pred_multi = int(nums[0]) if nums else 0
            # 边界修正
            if pred_multi > 5: pred_multi = 5
            if pred_multi < 0: pred_multi = 0
            
            y_true_multi_list.append(true_multi)
            y_pred_multi_list.append(pred_multi)
            
            results.append({
                "dialogue_id": dialogue_id,
                "true_multi": true_multi,
                "pred_multi": pred_multi,
                "response": response
            })

        # --- 计算指标 ---
        if len(y_true_multi_list) > 0:
            # 1. 多分类指标 (Macro Average)
            acc_m = accuracy_score(y_true_multi_list, y_pred_multi_list)
            p_m, r_m, f1_m, _ = precision_recall_fscore_support(
                y_true_multi_list, y_pred_multi_list, average='macro', zero_division=0
            )
            
            metrics_multi["acc"].append(acc_m)
            metrics_multi["prec"].append(p_m)
            metrics_multi["rec"].append(r_m)
            metrics_multi["f1"].append(f1_m)
            
            # 2. 映射二分类指标 (Binary)
            y_true_bin = [map_to_binary(x) for x in y_true_multi_list]
            y_pred_bin = [map_to_binary(x) for x in y_pred_multi_list]
            
            acc_b = accuracy_score(y_true_bin, y_pred_bin)
            p_b, r_b, f1_b, _ = precision_recall_fscore_support(
                y_true_bin, y_pred_bin, average='binary', zero_division=0
            )
            
            metrics_bin["acc"].append(acc_b)
            metrics_bin["prec"].append(p_b)
            metrics_bin["rec"].append(r_b)
            metrics_bin["f1"].append(f1_b)

            # 保存详细结果
            with open(output_filename, 'w', encoding='utf-8') as f:
                for res in results:
                    f.write(json.dumps(res, ensure_ascii=False) + "\n")
            
            print(f"  -> Seed {seed} | Multi-F1: {f1_m:.4f} | Binary-F1: {f1_b:.4f}")

    # --- 汇总统计函数 ---
    def get_stats(metric_dict):
        stats = {}
        str_stats = {}
        raw_means = {}
        for k, v in metric_dict.items():
            if not v: continue
            m, s = np.mean(v), np.std(v)
            stats[k] = {"mean": m, "std": s}
            str_stats[k] = f"{m:.4f}±{s:.4f}"
            raw_means[k] = m
        return stats, str_stats, raw_means

    stats_multi, str_multi, means_multi = get_stats(metrics_multi)
    stats_bin, str_bin, means_bin = get_stats(metrics_bin)

    # 打印表格
    print("-" * 100)
    print(f"{'Type':<15} | {'Acc':<18} | {'Prec':<18} | {'Rec':<18} | {'F1':<18}")
    print("-" * 100)
    print(f"{'6-Class (Macro)':<15} | {str_multi.get('acc', 'N/A'):<18} | {str_multi.get('prec', 'N/A'):<18} | {str_multi.get('rec', 'N/A'):<18} | {str_multi.get('f1', 'N/A'):<18}")
    print(f"{'Binary (Mapped)':<15} | {str_bin.get('acc', 'N/A'):<18} | {str_bin.get('prec', 'N/A'):<18} | {str_bin.get('rec', 'N/A'):<18} | {str_bin.get('f1', 'N/A'):<18}")
    print("-" * 100)

    # 添加到最终结果
    final_stats_summary.append({
        "features": combo_key,
        "multi_class_metrics": stats_multi,
        "binary_mapped_metrics": stats_bin,
        # 方便排序用的字段
        "sort_score_bin_f1": means_bin.get("f1", 0),
        "sort_score_multi_f1": means_multi.get("f1", 0)
    })

# 5. 排序与保存
print("\n" + "="*40)
print("Final Summary (Sorted by Binary F1)")
print("="*40)

# 按二分类 F1 排序
final_stats_summary.sort(key=lambda x: x['sort_score_bin_f1'], reverse=True)

print(f"{'Features':<35} | {'Bin F1':<15} | {'Multi F1 (Macro)':<15}")
print("-" * 75)
for item in final_stats_summary:
    feat = item['features']
    bin_f1 = f"{item['binary_mapped_metrics']['f1']['mean']:.4f}"
    mul_f1 = f"{item['multi_class_metrics']['f1']['mean']:.4f}"
    print(f"{feat:<35} | {bin_f1:<15} | {mul_f1:<15}")

with open(summary_file, 'w', encoding='utf-8') as f:
    json.dump(final_stats_summary, f, indent=4, ensure_ascii=False)

print(f"\nSaved dual metrics to {summary_file}")