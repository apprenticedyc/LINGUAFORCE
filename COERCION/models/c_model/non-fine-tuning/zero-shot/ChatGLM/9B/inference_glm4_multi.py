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
output_file = "/root/autodl-tmp/non-fine-tuning/ChatGLM/zero_shot_multi_results.jsonl"

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
        true_multi = item.get("dialog_multi_label", -1)

        # 拼接对话上下文
        # dialogue_text = "\n".join([f"Utterance{j+1}: {utt}" for j, utt in enumerate(utterances)])
        non_empty_utts = [utt for utt in utterances if utt.strip() != ""]
        dialogue_text = "\n".join([f"Utterance{j+1}: {utt}" for j, utt in enumerate(non_empty_utts)])

        # 构造道德绑架的 6 分类 Prompt (结合了你之前定义的标签系)
        prompt_multi = (
            "You are an expert in moral coercion detection. Please read the following dialogue to assess the severity of your moral coercion and strictly output a numerical label between 0 and 5.\n"
            "Classification criteria:\n"
            "0: No coercion (normal communication) \n"
            "1: Minor suspicion (ambiguous)\n"
            "2: Slight bias (with slight pressure)\n"
            "3: Obvious suggestion (obvious moral pressure)\n"
            "4: Highly self-centered (ignoring others' feelings and forcefully exerting pressure)\n"
            "5: Extreme coercion (Strong Moral coercion and Accusation)\n\n"
            f"Dialogue Content: \n{dialogue_text}\n\n"
            "Your judgment (please only output a pure number between 0-5, do not explain):"
        )

        # 使用 GLM-4 专用的 chat template
        inputs = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_multi}],
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
        numbers = re.findall(r'\d+', response)
        if numbers:
            pred_multi = int(numbers[0])
            # 安全校验：确保提取的数字在合法范围内 (0, 1, 2, 3, 4, 5)
            if pred_multi not in [0, 1, 2, 3, 4, 5]: 
                pred_multi = -1 
        else:
            pred_multi = -1

        results.append({
            "dialogue_id": dialogue_id,
            "true_multi": true_multi,
            "pred_multi": pred_multi,
            "model_response": response,
            "utterances": utterances
        })

        print(f"Processed {i+1} | ID: {dialogue_id} | True: {true_multi} | Pred: {pred_multi} | Raw Reply: {response}")

# ================= 6. 统计准确率与保存 =================
valid_results = [r for r in results if r["pred_multi"] != -1]
correct = sum(1 for r in valid_results if r["pred_multi"] == r["true_multi"])
total_valid = len(valid_results)
total_all = len(results)

if total_valid > 0:
    accuracy = correct / total_valid
    print(f"\n[Multi-class] Accuracy on {total_valid} valid samples (out of {total_all}): {accuracy:.4f}")
else:
    print("\nNo valid predictions to compute accuracy.")

# 写入文件
with open(output_file, 'w', encoding='utf-8') as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"[*] Results successfully saved to {output_file}")