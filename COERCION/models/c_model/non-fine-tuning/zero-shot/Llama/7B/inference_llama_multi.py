import torch
import json
import re
import os
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ================= 1. 路径与模型配置 =================
model_name = "/root/autodl-tmp/Llama-2-7b-chat-ms"

print("[*] Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

print("[*] Loading model in bfloat16...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
model.eval()

# ================= 2. 数据与输出路径 =================
data_file = "/root/autodl-tmp/DATA/test.jsonl"
output_file = "/root/autodl-tmp/non-fine-tuning/zero_shot_multi_results.jsonl"
os.makedirs(os.path.dirname(output_file), exist_ok=True)

# 建议先用 10 条数据跑通测试，确认没问题后再改为 99999
max_samples = 10 
results = []

print(f"[*] Starting Zero-Shot inference on {data_file}...")

with open(data_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

for i, line in enumerate(tqdm(lines[:max_samples], desc="Inferencing")):
    item = json.loads(line.strip())
    dialogue_id = item.get("dialogue_id", f"unknown_{i}")
    utterances = item["utterances"]
    true_multi = item["dialog_multi_label"]

    # 清洗并构建对话文本
    non_empty_utts = [utt for utt in utterances if utt.strip() != ""]
    dialogue_text = "\n".join([f"Utterance{j+1}: {utt}" for j, utt in enumerate(non_empty_utts)])

    # ================= 3. 构建强约束 Prompt =================
    prompt_multi = (
        "Below is a conversation. Please assess the severity of moral coercion "
        "(0 indicates no moral coercion, 5 indicates extreme moral coercion).\n"
        "0: No moral coercion, 1: Slight suspicion, 2: Slight tendency towards moral coercion, "
        "3: Clear implication, 4: Highly self-centered, 5: Extreme moral coercion.\n"
        f"Conversation:\n{dialogue_text}\n\n"
        "Your judgment (CRITICAL: Output ONLY a single integer between 0 and 5, with no other text or explanation):"
    )

    system_prompt = "You are a helpful assistant that analyzes conversations."

    # Llama-2 官方指令格式
    text = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{prompt_multi} [/INST]"
    
    model_inputs = tokenizer(text, return_tensors="pt").to(model.device)

    # ================= 4. 模型生成 =================
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=100,  # 增加生成长度，容忍模型的“废话”
            do_sample=False,     # 贪心解码保证确定性
            pad_token_id=tokenizer.eos_token_id
        )

    # 截断 Prompt，提取回复
    input_length = model_inputs.input_ids.shape[1]
    response_ids = generated_ids[0][input_length:]
    response = tokenizer.decode(response_ids, skip_special_tokens=True).strip()

    # ================= 5. 结果解析与 Debug =================
    numbers = re.findall(r'\d+', response)
    if numbers:
        pred_multi = int(numbers[0])
        if pred_multi not in range(6):  
            pred_multi = -1  
    else:
        pred_multi = -1

    # 如果解析失败，打印出模型到底生成了什么，方便排查
    if pred_multi == -1:
        print(f"\n[Debug] Regex failed for ID {dialogue_id}. Model raw response: {repr(response)}")

    results.append({
        "dialogue_id": dialogue_id,
        "true_multi": true_multi,
        "pred_multi": pred_multi,
        "model_response": response,
        "utterances": utterances
    })

# ================= 6. 统计与保存 =================
valid_results = [r for r in results if r["pred_multi"] != -1]
correct = sum(1 for r in valid_results if r["pred_multi"] == r["true_multi"])
total_valid = len(valid_results)
total_all = len(results)

print("\n" + "="*40)
if total_valid > 0:
    accuracy = correct / total_valid
    print(f"Accuracy on {total_valid} valid samples (out of {total_all} total): {accuracy:.4f}")
    print(f"Failed parsing count: {total_all - total_valid}")
else:
    print("No valid predictions to compute accuracy.")
print("="*40)

with open(output_file, 'w', encoding='utf-8') as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
        
print(f"[*] Results saved to {output_file}")