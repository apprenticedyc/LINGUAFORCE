import json
import random
import numpy as np
import collections
import re
import os
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# ================= 1. 全局配置 =================
MODEL_NAME = "/root/autodl-tmp/GLM-4-32B-Chat" 
DATA_FILE = "/root/autodl-tmp/DATA/test.jsonl"
OUTPUT_FILE = "/root/autodl-tmp/non-fine-tuning/ChatGLM/few_shot_multi_and_binary_metrics.jsonl"

# 实验变量
SHOT_LIST = [0, 1, 2, 3, 4, 5, 6]
SEEDS = [42, 519, 2026]
MAX_TEST_SAMPLES = 99999

# ================= 2. 完整案例池 =================
# 注意：正式运行时请确保填充你的 Few-shot 数据池
FULL_EXAMPLE_POOL = []

# ================= 3. 辅助函数 =================

def get_balanced_multi_shots(pool, k, seed):
    """多分类轮询采样: 确保类别尽可能均匀"""
    if k == 0: return []
    rng = random.Random(seed)
    groups = collections.defaultdict(list)
    for item in pool: groups[item['label_m']].append(item)
    classes = list(range(6)) 
    selected = []
    while len(selected) < k:
        rng.shuffle(classes)
        for c in classes:
            if len(selected) >= k: break
            if groups[c]:
                choice = rng.choice(groups[c])
                selected.append(choice)
    rng.shuffle(selected)
    return selected

def format_dialogue(utterance_list):
    valid_utts = [u for u in utterance_list if u.strip()]
    return "\n".join([f"Utterance{i+1}: {u}" for i, u in enumerate(valid_utts)])

def build_prompt(dialogue_str):
    return (
        "Below is a conversation. Please assess the severity of moral moral coercion "
        "(0 indicates no moral coercion, 5 indicates extreme moral coercion).\n"
        "Severity Levels:\n"
        "0: No moral coercion\n"
        "1: Slight suspicion\n"
        "2: Slight tendency\n"
        "3: Clear implication\n"
        "4: Highly self-centered\n"
        "5: Extreme moral coercion\n\n"
        f"Conversation:\n{dialogue_str}\n\n"
        "Your judgment (output only a single number between 0 and 5):"
    )

def map_to_binary(label_score):
    # 0, 1 -> 0 (Non-Coercion)
    # 2, 3, 4, 5 -> 1 (Coercion)
    return 0 if label_score <= 1 else 1

# ================= 4. vLLM 引擎初始化 =================

print(f"[*] Loading GLM-4-32B Model from {MODEL_NAME} with vLLM (4x5090 config)...")

# 专为 4x5090 打造的满血张量并行配置
llm = LLM(
    model=MODEL_NAME, 
    tensor_parallel_size=4,      # 4 张 5090 齐上阵，完美整除 65024
    dtype="bfloat16",            # 满血半精度
    trust_remote_code=True,      # 加载 GLM 的自定义分词器
    gpu_memory_utilization=0.90, # 显存极大，留 10% 缓冲
    max_model_len=8192,          # 解锁 8192 长上下文
    enforce_eager=True           # 防止 CUDA Graph 在多卡下死锁
)
tokenizer = llm.get_tokenizer()

# 在 vLLM 中，temperature=0.0 等价于贪婪解码（do_sample=False）
sampling_params = SamplingParams(
    temperature=0.0, 
    max_tokens=5
)

# 预加载数据
test_data = []
try:
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= MAX_TEST_SAMPLES: break
            test_data.append(json.loads(line.strip()))
    print(f"Loaded {len(test_data)} test samples.")
except FileNotFoundError:
    print(f"Error: Data file not found at {DATA_FILE}")
    exit()

# 初始化统计容器
stats_multi = {k: {'acc': [], 'f1': [], 'prec': [], 'rec': []} for k in SHOT_LIST}
stats_bin_agg = {k: {'acc': [], 'f1': [], 'prec': [], 'rec': []} for k in SHOT_LIST}

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

# ================= 5. 主实验流程 =================

for k in SHOT_LIST:
    print(f"\n{'='*20} Testing {k}-Shot {'='*20}")
    
    for seed in SEEDS:
        # 1. 采样与 Base Prompt 构造
        current_shots = get_balanced_multi_shots(FULL_EXAMPLE_POOL, k=k, seed=seed)
        
        # GLM-4 System Prompt
        base_messages = [{"role": "system", "content": "You are a helpful assistant that analyzes conversations for moral pressure."}]
        
        # 构建 Few-shot 历史
        for shot in current_shots:
            shot_text = format_dialogue(shot["utts"])
            base_messages.append({"role": "user", "content": build_prompt(shot_text)})
            base_messages.append({"role": "assistant", "content": str(shot["label_m"])})
            
        prompts = []
        y_true_multi = []
        
        # 2. 组装批量推理请求
        for item in test_data:
            true_label = item.get("dialog_multi_label", -1) 
            if true_label == -1: continue # 跳过无标签数据
            
            target_text = format_dialogue(item["utterances"])
            messages = base_messages.copy()
            messages.append({"role": "user", "content": build_prompt(target_text)})
            
            # 直接渲染为纯文本交给 vLLM
            text = tokenizer.apply_chat_template(
                messages, 
                add_generation_prompt=True, 
                tokenize=False
            )
            prompts.append(text)
            y_true_multi.append(true_label)
            
        # 3. vLLM 极速并发推理
        print(f"  [Seed {seed}] Running vLLM inference for {len(prompts)} samples...")
        outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
        
        # 4. 结果解析
        y_pred_multi = []
        for output in outputs:
            response = output.outputs[0].text.strip()
            nums = re.findall(r'\d+', response)
            pred = int(nums[0]) if nums else 0 # 默认 fallback
            if pred > 5: pred = 5 
            y_pred_multi.append(pred)
            
        # 5. 计算多分类指标 (6-Class)
        if len(y_true_multi) > 0:
            p_m, r_m, f1_m, _ = precision_recall_fscore_support(y_true_multi, y_pred_multi, average='macro', zero_division=0)
            acc_m = accuracy_score(y_true_multi, y_pred_multi)
            
            stats_multi[k]['acc'].append(acc_m)
            stats_multi[k]['f1'].append(f1_m)
            stats_multi[k]['prec'].append(p_m)
            stats_multi[k]['rec'].append(r_m)
            
            # 6. 计算聚合二分类指标 (Binary Aggregated)
            y_true_bin = [map_to_binary(y) for y in y_true_multi]
            y_pred_bin = [map_to_binary(y) for y in y_pred_multi]
            
            p_b, r_b, f1_b, _ = precision_recall_fscore_support(y_true_bin, y_pred_bin, average='binary', zero_division=0)
            acc_b = accuracy_score(y_true_bin, y_pred_bin)
            
            stats_bin_agg[k]['acc'].append(acc_b)
            stats_bin_agg[k]['f1'].append(f1_b)
            stats_bin_agg[k]['prec'].append(p_b)
            stats_bin_agg[k]['rec'].append(r_b)
            
            print(f"  -> Seed {seed} | [Multi] F1: {f1_m:.4f} | [Binary-Agg] F1: {f1_b:.4f}")
        else:
            print(f"  -> Seed {seed} | No valid labels found.")

# ================= 6. 打印双重报告 =================

def print_table(title, stats_dict):
    print("\n" + "="*80)
    print(f"{title:^80}")
    print("="*80)
    print(f"{'K':<3} | {'Acc (Mean±Std)':<18} | {'Prec (Mean±Std)':<18} | {'Rec (Mean±Std)':<18} | {'F1 (Mean±Std)':<18}")
    print("-" * 80)
    
    results = []
    for k in SHOT_LIST:
        accs, f1s = stats_dict[k]['acc'], stats_dict[k]['f1']
        precs, recs = stats_dict[k]['prec'], stats_dict[k]['rec']
        
        if not accs: 
            m_acc, s_acc = 0.0, 0.0
            m_f1, s_f1 = 0.0, 0.0
            m_p, s_p = 0.0, 0.0
            m_r, s_r = 0.0, 0.0
        else:
            m_acc, s_acc = np.mean(accs)*100, np.std(accs)*100
            m_f1, s_f1 = np.mean(f1s)*100, np.std(f1s)*100
            m_p, s_p = np.mean(precs)*100, np.std(precs)*100
            m_r, s_r = np.mean(recs)*100, np.std(recs)*100
        
        print(f"{k:<3} | {m_acc:.1f}±{s_acc:.1f}%".ljust(21) + f" | {m_p:.1f}±{s_p:.1f}%".ljust(21) + 
              f" | {m_r:.1f}±{s_r:.1f}%".ljust(21) + f" | {m_f1:.1f}±{s_f1:.1f}%")
        
        results.append({
            "k": k, "acc": f"{m_acc:.2f}±{s_acc:.2f}", "f1": f"{m_f1:.2f}±{s_f1:.2f}",
            "prec": f"{m_p:.2f}±{s_p:.2f}", "rec": f"{m_r:.2f}±{s_r:.2f}"
        })
    return results

# 打印并保存
r_multi = print_table("GLM-4-32B | 6-CLASS CLASSIFICATION REPORT (Macro Avg)", stats_multi)
r_bin = print_table("GLM-4-32B | BINARY AGGREGATED REPORT (0-1=Neg, 2-5=Pos)", stats_bin_agg)

# 组合保存
final_output = []
for i, k in enumerate(SHOT_LIST):
    final_output.append({
        "k_shot": k,
        "multi_class_metrics": r_multi[i],
        "binary_agg_metrics": r_bin[i]
    })

with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    for item in final_output:
        f.write(json.dumps(item) + "\n")
        
print(f"\nAll metrics saved to {OUTPUT_FILE}")