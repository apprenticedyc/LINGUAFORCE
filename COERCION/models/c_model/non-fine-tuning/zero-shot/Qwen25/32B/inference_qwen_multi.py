import json
import re
import os
from sklearn.metrics import accuracy_score, f1_score, classification_report
from vllm import LLM, SamplingParams

# ================= 1. 配置区域 =================
# 模型与数据路径
MODEL_PATH = "/root/autodl-tmp/Qwen25-32B-Instruct" 
DATA_FILE = "/root/autodl-tmp/DATA/test.jsonl"
OUTPUT_DIR = "/root/autodl-tmp/non-fine-tuning/Qwen25_32B"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "zero_shot_multi_results.jsonl")

# 最大测试样本数 (None 表示跑全量)
MAX_SAMPLES = None 

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================= 2. 初始化 vLLM =================
print(f"Loading model: {MODEL_PATH} with vLLM across 3 GPUs...")

# 沿用 3x5090 的最佳流水线并行配置
llm = LLM(
    model=MODEL_PATH, 
    tensor_parallel_size=1,      # 不切分注意力头
    pipeline_parallel_size=3,    # 纵向切分层，分配到 3 张卡
    dtype="bfloat16",         
    trust_remote_code=True,
    gpu_memory_utilization=0.9,
    max_model_len=8192,          # 限制显存预分配
    enforce_eager=True           # 禁用 CUDA 图，防止多卡通信死锁
)

# 获取自带的 tokenizer 用于套用 Chat Template
tokenizer = llm.get_tokenizer()

# 对应原代码的贪心解码配置 (do_sample=False -> temperature=0.0)
sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=10
)

# ================= 3. 数据准备与 Prompt 构建 =================
print(f"Loading data from {DATA_FILE}...")
prompts = []
metadata = []

with open(DATA_FILE, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if MAX_SAMPLES and i >= MAX_SAMPLES:
            break

        item = json.loads(line.strip())
        dialogue_id = item["dialogue_id"]
        utterances = item["utterances"]
        true_multi = item["dialog_multi_label"]

        # 清理空句子并加上序号
        non_empty_utts = [utt for utt in utterances if utt.strip() != ""]
        dialogue_text = "\n".join([f"Utterance{j+1}: {utt}" for j, utt in enumerate(non_empty_utts)])

        # 构造 Prompt
        prompt_multi = (
            "Below is a conversation. Please assess the severity of moral moral coercion "
            "(0 indicates no moral coercion, 5 indicates extreme moral coercion).\n"
            "0: No moral coercion, 1: Slight suspicion, 2: Slight tendency towards moral moral coercion, "
            "3: Clear implication, 4: Highly self-centered, 5: Extreme moral coercion.\n"
            f"Conversation:\n{dialogue_text}\n\n"
            "Your judgment (output only a number between 0 and 5):"
        )

        messages = [
            {"role": "system", "content": "You are a helpful assistant that analyzes conversations."},
            {"role": "user", "content": prompt_multi}
        ]

        # 套用 Qwen 的对话模板
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        prompts.append(text)
        metadata.append({
            "dialogue_id": dialogue_id,
            "true_multi": true_multi,
            "utterances": utterances
        })

# ================= 4. vLLM 批量推理 =================
print(f"Starting inference for {len(prompts)} samples...")
# 一次性丢给 vLLM，引擎会自动进行 Continuous Batching 加速
outputs = llm.generate(prompts, sampling_params, use_tqdm=True)

# ================= 5. 结果解析与指标计算 =================
results = []
valid_y_true = []
valid_y_pred = []

for i, output in enumerate(outputs):
    response = output.outputs[0].text.strip()
    
    # 正则提取第一个独立数字
    numbers = re.findall(r'\b\d+\b', response)
    if numbers:
        pred_multi = int(numbers[0])
        # 检查是否在合法范围内 0-5
        if pred_multi not in range(6):
            pred_multi = -1
    else:
        pred_multi = -1
        
    meta = metadata[i]
    results.append({
        "dialogue_id": meta["dialogue_id"],
        "true_multi": meta["true_multi"],
        "pred_multi": pred_multi,
        "model_response": response,
        "utterances": meta["utterances"]
    })
    
    if pred_multi != -1:
        valid_y_true.append(meta["true_multi"])
        valid_y_pred.append(pred_multi)

# 写入文件
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

# 统计打印
total_valid = len(valid_y_pred)
total_all = len(results)

print("\n" + "="*50)
print("EXPERIMENT RESULTS")
print("="*50)
print(f"Total Samples: {total_all}")
print(f"Valid Parsed Samples: {total_valid} (Failed to parse: {total_all - total_valid})")

if total_valid > 0:
    acc = accuracy_score(valid_y_true, valid_y_pred)
    macro_f1 = f1_score(valid_y_true, valid_y_pred, average='macro')
    
    print(f"\nOverall Accuracy : {acc:.4f}")
    print(f"Macro F1 Score   : {macro_f1:.4f}")
    
    print("\nDetailed Classification Report:")
    # 打印每个类别的精确率、召回率、F1
    print(classification_report(valid_y_true, valid_y_pred, digits=4, zero_division=0))
    
print(f"\nResults saved to {OUTPUT_FILE}")