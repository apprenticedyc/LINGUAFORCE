import torch
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ================= 1. 路径配置 =================
model_path = "/root/autodl-tmp/Qwen2.5-14B-Instruct"  # 原始权重
lora_path = "/root/autodl-tmp/fine-tuning/Qwen25_LoRA/best_lora"  # 训练好的权重
test_data_path = "/root/autodl-tmp/DATA/test.jsonl"  # 测试集路径
output_json = "/root/autodl-tmp/fine-tuning/Qwen25_LoRA/results.jsonl"

# ================= 2. 加载模型与 Tokenizer =================
print("[*] Loading base model and LoRA weights...")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# 以 4-bit 加载基础模型（节省显存）或 bfloat16 加载（速度更快）
base_model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

# 加载 LoRA 适配器
model = PeftModel.from_pretrained(base_model, lora_path)
model.eval() # 切换到推理模式

# ================= 3. 推理函数 =================
def predict_severity(utterances):
    # 保持与训练时完全一致的清洗逻辑
    non_empty_utts = [utt for utt in utterances if utt.strip() != ""]
    dialogue_text = "\n".join([f"Utterance{j+1}: {utt}" for j, utt in enumerate(non_empty_utts)])

    # 保持与训练时完全一致的 Prompt
    prompt_multi = (
        "Below is a conversation. Please assess the severity of moral blackmail "
        "(0 indicates no blackmail, 5 indicates extreme blackmail).\n"
        "0: No blackmail, 1: Slight suspicion, 2: Slight tendency towards moral blackmail, "
        "3: Clear implication, 4: Highly self-centered, 5: Extreme blackmail.\n"
        f"Conversation:\n{dialogue_text}\n\n"
        "Your judgment (output only a number between 0 and 5):"
    )

    messages = [
        {"role": "system", "content": "You are a helpful assistant that analyzes conversations."},
        {"role": "user", "content": prompt_multi}
    ]

    # 使用 apply_chat_template，注意推理时 add_generation_prompt=True
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=10, # 我们只需要输出一个数字，10个 token 足够了
            do_sample=False,   # 设为 False 保证输出的确定性
            pad_token_id=tokenizer.eos_token_id
        )
    
    # 只取生成的回复部分
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, outputs)]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response.strip()

# ================= 4. 执行批量测试 =================
results = []
print(f"[*] Starting inference on {test_data_path}...")

with open(test_data_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

for line in tqdm(lines):
    data = json.loads(line)
    prediction = predict_severity(data["utterances"])
    
    # 保存结果以便后续对比
    results.append({
        "dialog_id": data.get("dialogue_id", "N/A"),
        "ground_truth": str(data["dialog_multi_label"]),
        "prediction": prediction
    })

# 保存到 JSONL
with open(output_json, "w", encoding="utf-8") as f:
    for res in results:
        f.write(json.dumps(res, ensure_ascii=False) + "\n")

print(f"[*] Inference finished. Results saved to {output_json}")