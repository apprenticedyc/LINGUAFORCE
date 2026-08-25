import json
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. 加载模型和分词器（与原始代码相同）
model_name = "/root/autodl-tmp/Qwen2.5-14B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 2. 数据集路径（请根据实际情况修改）
data_file = "/root/autodl-tmp/DATA/test.jsonl"

# 可选：限制测试的样本数量（例如只测试前10条）
max_samples = 99999

# 用于存储结果的列表
results = []

# 3. 逐行处理数据集
with open(data_file, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if max_samples and i >= max_samples:
            break
        
        # 解析JSON
        item = json.loads(line.strip())
        dialogue_id = item["dialogue_id"]
        utterances = item["utterances"]
        true_binary = item["dialog_binary_label"]
        true_multi = item["dialog_multi_label"]
        
        # 将对话列表转换为一个连贯的文本（例如每句话用换行或标点分隔）
        # dialogue_text = "\n".join([f"第{j+1}句：{utt}" for j, utt in enumerate(utterances)])
        non_empty_utts = [utt for utt in utterances if utt.strip() != ""]
        dialogue_text = "\n".join([f"Utterance{j+1}: {utt}" for j, utt in enumerate(non_empty_utts)])

        # 4. 构造提示
        # 二分类任务：输出0或1
        prompt_binary = f"Below is a conversation. Please determine whether moral blackmail exists. Output 1 if it exists; output 0 if it does not.\nConversation:\n{dialogue_text}\n\nYour judgment (output only the number 0 or 1):"
        
        # 这里我们演示二分类任务，如果要测试多分类，将 prompt_binary 替换为 prompt_multi
        prompt = prompt_binary
        
        # 按照Qwen对话模板构造消息
        messages = [
            {"role": "system", "content": "You are a helpful assistant that analyzes conversations."},
            {"role": "user", "content": prompt}
        ]
        
        # 应用模板并编码
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        # 5. 生成回答（限制输出长度，因为只需一个数字）
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=10,          # 最多生成10个token，足够输出数字
            do_sample=False,             # 使用贪心解码，减少随机性
            pad_token_id=tokenizer.eos_token_id  # 避免警告
        )
        
        # 只保留新生成的部分
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        
        # 6. 解析模型输出，尝试提取数字
        import re
        numbers = re.findall(r'\d+', response)  # 提取所有连续数字
        if numbers:
            pred_binary = int(numbers[0])       # 取第一个数字
            # 确保预测值在合理范围内（二分类只能0或1）
            if pred_binary not in [0, 1]:
                pred_binary = -1  # 表示解析失败
        else:
            pred_binary = -1
        
        # 7. 记录结果
        results.append({
            "dialogue_id": dialogue_id,
            "true_binary": true_binary,
            "pred_binary": pred_binary,
            "model_response": response,
            "utterances": utterances  # 可选，便于查看
        })
        
        # 打印进度
        print(f"Processed {i+1} samples. ID: {dialogue_id}, True: {true_binary}, Pred: {pred_binary}")

# 8. 统计准确率（简单示例）
correct = sum(1 for r in results if r["pred_binary"] == r["true_binary"])
total = len(results)
if total > 0:
    accuracy = correct / total
    print(f"\nAccuracy on {total} samples: {accuracy:.2f}")

# 9. 保存详细结果到文件
output_file = "/root/autodl-tmp/non-fine-tuning/Qwen25/zero_shot_results.jsonl"
with open(output_file, 'w', encoding='utf-8') as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"Results saved to {output_file}")