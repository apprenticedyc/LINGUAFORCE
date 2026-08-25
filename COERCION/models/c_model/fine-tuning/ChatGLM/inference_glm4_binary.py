import json
import re
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

os.environ["OMP_NUM_THREADS"] = "1"

# ================= 1. 配置路径与设备 =================
base_model_path = "/root/autodl-tmp/glm-4-9b-chat"
lora_weights_path = "/root/autodl-tmp/fine-tuning/TalkDown/binary/glm4-lora-talkdown-binary" # 你的二分类 LoRA 权重路径
test_data_file = "/root/autodl-tmp/DATA/test.jsonl"  # 替换为真实的测试集路径
output_file = "/root/autodl-tmp/fine-tuning/TalkDown/binary/talkdown_binary_results.jsonl"

device = "cuda" if torch.cuda.is_available() else "cpu"

# ================= 2. 加载基础模型与 LoRA 权重 =================
print(f"[*] Loading Tokenizer from {base_model_path}...")
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

print(f"[*] Loading Base Model in bfloat16...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    device_map="auto"
)

# 【核心步骤】：将 LoRA 权重合并到基础模型上
print(f"[*] Loading and merging LoRA weights from {lora_weights_path}...")
model = PeftModel.from_pretrained(base_model, lora_weights_path)
model.eval() # 切换到评估模式，关闭 Dropout 等训练特性

# ================= 3. 开始批量推理 =================
results = []
max_samples = None  # 如果你想先测 10 条看看效果，可以改为 10

# 记录统计信息
valid_results = 0
correct_predictions = 0

print("[*] Starting Inference...")
with open(test_data_file, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if max_samples and i >= max_samples:
            break

        item = json.loads(line.strip())
        dialogue_id = item.get("dialogue_id", f"idx_{i}")
        utterances = item.get("utterances", [])
        true_binary = item.get("dialog_binary_label", -1)

        # 过滤空句子并拼接
        non_empty_utts = [utt for utt in utterances if utt.strip() != ""]
        dialogue_text = "\n".join([f"Utterance{j+1}: {utt}" for j, utt in enumerate(non_empty_utts)])

        # 【重点】：这里的 Prompt 必须和训练时一字不差！
        prompt_binary = f"Below is a conversation. Please determine whether moral blackmail exists. Output 1 if it exists; output 0 if it does not.\nConversation:\n{dialogue_text}\n\nYour judgment (output only the number 0 or 1):"

        inputs = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_binary}],
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True
        ).to(model.device)

        # ================= 4. 模型生成 =================
        gen_kwargs = {
            "max_new_tokens": 5,           # 分类任务只需要极短的输出
            "do_sample": False,            # 关闭采样，使用贪心解码保证稳定输出
            "pad_token_id": tokenizer.eos_token_id 
        }
        
        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
            # 截取新生成的部分 (排除掉输入的 Prompt)
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
            response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        # ================= 5. 解析结果 =================
        numbers = re.findall(r'\d+', response)
        if numbers:
            pred_binary = int(numbers[0])
            if pred_binary not in [0, 1]:
                pred_binary = -1
        else:
            pred_binary = -1

        results.append({
            "dialogue_id": dialogue_id,
            "true_binary": true_binary,
            "pred_binary": pred_binary,
            "model_response": response,
            "utterances": utterances
        })

        # 实时计算当前准确率
        if pred_binary != -1 and true_binary in [0, 1]:
            valid_results += 1
            if pred_binary == true_binary:
                correct_predictions += 1

        print(f"Processed {i+1} | ID: {dialogue_id} | True: {true_binary} | Pred: {pred_binary} | Raw Reply: {response}")

# ================= 6. 统计与保存 =================
if valid_results > 0:
    accuracy = correct_predictions / valid_results
    print(f"\n[Binary-class LoRA] Accuracy on {valid_results} valid samples: {accuracy:.4f}")
else:
    print("\nNo valid predictions to compute accuracy.")

# 确保输出目录存在
os.makedirs(os.path.dirname(output_file), exist_ok=True)

with open(output_file, 'w', encoding='utf-8') as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"[*] Results successfully saved to {output_file}")