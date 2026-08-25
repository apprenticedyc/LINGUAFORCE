import json
import re
import os
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

# ================= 1. 配置区域 =================
# 基础模型路径 (32B)
BASE_MODEL_PATH = "/root/autodl-tmp/Qwen2.5-32B-Instruct" 
# 微调得到的 LoRA 权重路径
LORA_PATH = "/root/autodl-tmp/fine-tuning/binary/Qwen25_32B_QLoRA/best_lora"
# 测试集数据路径
DATA_FILE = "/root/autodl-tmp/DATA/test.jsonl"
# 结果保存路径
OUTPUT_DIR = "/root/autodl-tmp/fine-tuning/binary/Qwen25_32B_QLoRA_eval"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "lora_test_results.jsonl")

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================= 2. 初始化 vLLM (开启 LoRA) =================
print(f"Loading Base Model: {BASE_MODEL_PATH} with vLLM across 3 GPUs...")

# llm = LLM(
#     model=BASE_MODEL_PATH, 
#     tensor_parallel_size=1,      
#     pipeline_parallel_size=3,    # 纵向切分，利用 3 张 5090
#     dtype="bfloat16",         
#     trust_remote_code=True,
#     gpu_memory_utilization=0.9,
#     max_model_len=8192,          
#     enforce_eager=True,          # 防止多卡死锁
#     enable_lora=True,            # 开启 LoRA 支持
#     max_lora_rank=16             # 必须与你训练时设置的 r 保持一致或更大
# )

llm = LLM(
    model=BASE_MODEL_PATH, 
    tensor_parallel_size=1,      
    pipeline_parallel_size=3,    
    dtype="bfloat16",         
    trust_remote_code=True,
    
    # === 核心修改点 ===
    gpu_memory_utilization=0.85, # 从 0.9 降到 0.85，给 LoRA 和 KV Cache 留出更多呼吸空间
    max_model_len=4096,          # 从 8192 降到 4096 (既然是短对话，4096 绝对够用)
    max_num_seqs=32,             # 限制单次批处理的并发数，防止它一口吞太多噎住卡死
    # =================
    
    enforce_eager=True,          
    enable_lora=True,            
    max_lora_rank=16             
)

tokenizer = llm.get_tokenizer()

# 贪婪解码，保证测试结果稳定
sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=5
)

# 实例化 LoRA 请求
lora_request = LoRARequest("my_binary_lora", 1, LORA_PATH)

# ================= 3. 数据准备与 Prompt 构建 =================
print(f"Loading data from {DATA_FILE}...")
prompts = []
metadata = []

with open(DATA_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        if not line.strip(): continue
        
        item = json.loads(line.strip())
        dialogue_id = item.get("dialogue_id", item.get("id", "unknown"))
        utterances = item["utterances"]
        true_binary = item["dialog_binary_label"]

        # 清理空句子并加上序号 (严格对齐训练集的数据清洗逻辑)
        non_empty_utts = [utt for utt in utterances if utt.strip() != ""]
        dialogue_text = "\n".join([f"Utterance{j+1}: {utt}" for j, utt in enumerate(non_empty_utts)])

        # 严格对齐微调训练时的 Prompt 模板
        prompt_binary = (
            "Below is a conversation. Please determine whether moral blackmail exists. "
            "Output 1 if it exists; output 0 if it does not.\n"
            f"Conversation:\n{dialogue_text}\n\n"
            "Your judgment (output only the number 0 or 1):"
        )

        messages = [
            {"role": "system", "content": "You are a helpful assistant that analyzes conversations."},
            {"role": "user", "content": prompt_binary}
        ]

        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        prompts.append(text)
        metadata.append({
            "dialogue_id": dialogue_id,
            "true_binary": true_binary,
            "utterances": utterances
        })

# ================= 4. vLLM 带 LoRA 批量推理 =================
print(f"Starting inference for {len(prompts)} samples with LoRA...")
# 传入 lora_request 让 vLLM 在生成时动态挂载 LoRA 权重
outputs = llm.generate(
    prompts, 
    sampling_params, 
    use_tqdm=True, 
    lora_request=lora_request
)

# ================= 5. 结果解析与指标计算 =================
results = []
valid_y_true = []
valid_y_pred = []

for i, output in enumerate(outputs):
    response = output.outputs[0].text.strip()
    
    # 正则提取结果
    numbers = re.findall(r'\b(0|1)\b', response)
    pred_binary = int(numbers[0]) if numbers else -1
        
    meta = metadata[i]
    results.append({
        "dialogue_id": meta["dialogue_id"],
        "true_binary": meta["true_binary"],
        "pred_binary": pred_binary,
        "model_response": response
    })
    
    if pred_binary != -1:
        valid_y_true.append(meta["true_binary"])
        valid_y_pred.append(pred_binary)

# 保存推理详情
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

# 计算并打印评估指标
total_valid = len(valid_y_pred)
total_all = len(results)

print("\n" + "="*50)
print("FINETUNED LORA EXPERIMENT RESULTS")
print("="*50)
print(f"Total Samples: {total_all}")
print(f"Valid Parsed : {total_valid} (Failed: {total_all - total_valid})")

if total_valid > 0:
    acc = accuracy_score(valid_y_true, valid_y_pred)
    p, r, f1, _ = precision_recall_fscore_support(valid_y_true, valid_y_pred, average='binary', zero_division=0)
    
    print(f"\nAccuracy  : {acc:.4f}")
    print(f"Precision : {p:.4f}")
    print(f"Recall    : {r:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    
print(f"\nDetailed results saved to {OUTPUT_FILE}")