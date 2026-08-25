import json
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import os
os.environ["OMP_NUM_THREADS"] = "1"

# ================= 1. 配置路径与设备 =================
# 注意 Windows 下的路径斜杠，建议使用绝对路径
model_name = "/root/autodl-tmp/glm-4-9b-chat"
data_file = "/root/autodl-tmp/DATA/test.jsonl"  # 替换为你的真实测试集路径
output_file = "/root/autodl-tmp/non-fine-tuning/ChatGLM/zero_shot_binary_results.jsonl"

device = "cuda" if torch.cuda.is_available() else "cpu"

# ================= 2. 加载模型和分词器 =================
print(f"[*] Loading Tokenizer from {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

print(f"[*] Loading Model in bfloat16 to {device}...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16, # 如果是老显卡，请改为 torch.float16
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    device_map="auto"           # 自动分配显存，比硬编码 .to(device) 更安全
).eval()

# ================= 3. 开始批量推理 =================
results = []
max_samples = 99999  # 可改为 5 或 10 先跑个小批量测试

with open(data_file, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if max_samples and i >= max_samples:
            break

        item = json.loads(line.strip())
        dialogue_id = item.get("dialogue_id", f"idx_{i}")
        utterances = item.get("utterances", [])
        true_binary = item.get("dialog_binary_label", -1)

        # 拼接对话上下文
        # dialogue_text = "\n".join([f"Utterance{j+1}: {utt}" for j, utt in enumerate(utterances)])
        non_empty_utts = [utt for utt in utterances if utt.strip() != ""]
        dialogue_text = "\n".join([f"Utterance{j+1}: {utt}" for j, utt in enumerate(non_empty_utts)])

        # 构造道德绑架的 2 分类 Prompt (结合了你之前定义的标签系)
        prompt_binary = f"Below is a conversation. Please determine whether moral blackmail exists. Output 1 if it exists; output 0 if it does not.\nConversation:\n{dialogue_text}\n\nYour judgment (output only the number 0 or 1):"

        # 使用 GLM-4 专用的 chat template
        inputs = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_binary}],
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True
        ).to(model.device)

        # ================= 4. 模型生成 =================
        # 做分类任务时，建议关闭 do_sample（使用贪心解码），保证结果的确定性
        gen_kwargs = {
            "max_new_tokens": 10, # 我们只需要一个数字，不用生成太长
            "do_sample": False,   
            "pad_token_id": tokenizer.eos_token_id # 消除 pad 警告
        }
        
        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
            # 截取新生成的部分 (排除掉输入的 Prompt)
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
            response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        # ================= 5. 解析结果 =================
        # 用正则提取模型回答中的第一个数字
        numbers = re.findall(r'\d+', response)  # 提取所有连续数字
        if numbers:
            pred_binary = int(numbers[0])       # 取第一个数字
            # 确保预测值在合理范围内（二分类只能0或1）
            if pred_binary not in [0, 1]:
                pred_binary = -1  # 表示解析失败
        else:
            pred_binary = -1

        results.append({
            "dialogue_id": dialogue_id,
            "true_binary": true_binary,
            "pred_binary": pred_binary,
            "model_response": response,
            "utterances": utterances
        })

        print(f"Processed {i+1} | ID: {dialogue_id} | True: {true_binary} | Pred: {pred_binary} | Raw Reply: {response}")

# ================= 6. 统计准确率与保存 =================
valid_results = [r for r in results if r["pred_binary"] != -1]
correct = sum(1 for r in valid_results if r["pred_binary"] == r["true_binary"])
total_valid = len(valid_results)
total_all = len(results)

if total_valid > 0:
    accuracy = correct / total_valid
    print(f"\n[Binary-class] Accuracy on {total_valid} valid samples (out of {total_all}): {accuracy:.4f}")
else:
    print("\nNo valid predictions to compute accuracy.")

# 写入文件
with open(output_file, 'w', encoding='utf-8') as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"[*] Results successfully saved to {output_file}")