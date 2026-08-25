import torch
import json
import os
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ================= 1. 路径配置 =================
model_path = "/root/autodl-tmp/Llama-2-13b-chat-ms"  # 原始权重
lora_path = "/root/autodl-tmp/fine-tuning/binary/Llama13_LoRA/best_lora"  # 训练好的权重
test_data_path = "/root/autodl-tmp/DATA/test.jsonl"  # 测试集路径
output_json = "/root/autodl-tmp/fine-tuning/binary/Llama13_LoRA/results.jsonl"

# ================= 2. 加载模型与 Tokenizer =================
print("[*] Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
# 【修改】显式设定 Llama-2 的 pad_token
tokenizer.pad_token = tokenizer.eos_token
# 【修改】推理阶段如果要做 batching，必须用左侧填充 (Left Padding) 避免破坏生成位置
tokenizer.padding_side = "left"

print("[*] Loading base model in bfloat16...")
# 直接以 bfloat16 加载基础模型，不使用任何量化
base_model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

print("[*] Merging LoRA weights...")
# 加载 LoRA 适配器
model = PeftModel.from_pretrained(base_model, lora_path)
model.eval() # 切换到推理模式

# ================= 3. 推理函数 =================
def predict_severity(utterances):
    # 保持与训练时完全一致的清洗逻辑
    non_empty_utts = [utt for utt in utterances if utt.strip() != ""]
    dialogue_text = "\n".join([f"Utterance{j+1}: {utt}" for j, utt in enumerate(non_empty_utts)])

    # 保持与训练时完全一致的提示文本
    system_prompt = "You are a helpful assistant that analyzes conversations."
    prompt_binary = f"Below is a conversation. Please determine whether moral blackmail exists. Output 1 if it exists; output 0 if it does not.\nConversation:\n{dialogue_text}\n\nYour judgment (output only the number 0 or 1):"

    # 【修改】舍弃 apply_chat_template，直接使用 Llama-2 格式硬编码
    # 注意：推理时我们只给出用户的输入，不包含答案，并且以 [/INST] 结尾，引导模型开始生成
    text = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{prompt_binary} [/INST]"
    
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=10, # 我们只需要输出一个数字，10个 token 足够了
            do_sample=False,   # 设为 False 相当于 Greedy Search，保证输出的确定性
            pad_token_id=tokenizer.eos_token_id
        )
    
    # 截断 Prompt 部分，只取生成的回复
    input_length = inputs.input_ids.shape[1]
    generated_ids = outputs[0][input_length:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    return response.strip()

# ================= 4. 执行批量测试 =================
# 确保输出目录存在
os.makedirs(os.path.dirname(output_json), exist_ok=True)

results = []
print(f"[*] Starting inference on {test_data_path}...")

with open(test_data_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

for line in tqdm(lines):
    data = json.loads(line)
    prediction = predict_severity(data["utterances"])
    
    # 保存结果以便后续计算准确率、F1 等指标
    results.append({
        "dialogue_id": data.get("dialogue_id", "N/A"),
        "ground_truth": str(data["dialog_binary_label"]),
        "prediction": prediction
    })

# ================= 5. 保存结果 =================
with open(output_json, "w", encoding="utf-8") as f:
    for res in results:
        f.write(json.dumps(res, ensure_ascii=False) + "\n")

print(f"[*] Inference finished. Results saved to {output_json}")