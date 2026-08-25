import json
import re
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. 加载模型和分词器
model_name = "/root/autodl-tmp/Qwen2.5-14B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 2. 数据集路径
data_file = "/root/autodl-tmp/DATA/test.jsonl"

# 可选：限制测试样本数
max_samples = 99999  # 实际会处理全部

results = []

with open(data_file, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if max_samples and i >= max_samples:
            break

        item = json.loads(line.strip())
        dialogue_id = item["dialogue_id"]
        utterances = item["utterances"]
        true_multi = item["dialog_multi_label"]   # 真实多分类标签

        # 将对话转换为连贯文本（带序号）
        # dialogue_text = "\n".join([f"第{j+1}句：{utt}" for j, utt in enumerate(utterances)])
        non_empty_utts = [utt for utt in utterances if utt.strip() != ""]
        dialogue_text = "\n".join([f"Utterance{j+1}: {utt}" for j, utt in enumerate(non_empty_utts)])

        # 构造多分类提示（0-5，根据你的数据集定义调整描述）
        prompt_multi = (
            "Below is a conversation. Please assess the severity of moral moral coercion "
            "(0 indicates no moral coercion, 5 indicates extreme moral coercion).\n"
            "0: No moral coercion, 1: Slight suspicion, 2: Slight tendency towards moral moral coercion, "
            "3: Clear implication, 4: Highly self-centered, 5: Extreme moral coercion.\n"
            f"Conversation:\n{dialogue_text}\n\n"
            "Your judgment (output only a number between 0 and 5):"
        )

        # 按照Qwen对话模板构造消息
        messages = [
            {"role": "system", "content": "You are a helpful assistant that analyzes conversations."},
            {"role": "user", "content": prompt_multi}
        ]

        # 应用模板并编码
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        # 生成回答（限制长度，贪心解码）
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=10,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

        # 只保留新生成的部分
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        # 解析输出，提取第一个数字作为预测值
        numbers = re.findall(r'\d+', response)
        if numbers:
            pred_multi = int(numbers[0])
            # 根据你的实际标签范围调整检查条件（这里假设0-5）
            if pred_multi not in range(6):  # 0-5 共6个值
                pred_multi = -1  # 超出范围，视为解析失败
        else:
            pred_multi = -1

        # 记录结果
        results.append({
            "dialogue_id": dialogue_id,
            "true_multi": true_multi,
            "pred_multi": pred_multi,
            "model_response": response,
            "utterances": utterances
        })

        # 打印进度
        print(f"Processed {i+1} samples. ID: {dialogue_id}, True: {true_multi}, Pred: {pred_multi}")

# 统计准确率（仅计算预测成功的样本，-1表示解析失败）
valid_results = [r for r in results if r["pred_multi"] != -1]
correct = sum(1 for r in valid_results if r["pred_multi"] == r["true_multi"])
total_valid = len(valid_results)
total_all = len(results)

if total_valid > 0:
    accuracy = correct / total_valid
    print(f"\nAccuracy on {total_valid} valid samples (out of {total_all} total): {accuracy:.2f}")
else:
    print("\nNo valid predictions to compute accuracy.")

# 保存详细结果
output_file = "/root/autodl-tmp/non-fine-tuning/Qwen25/zero_shot_multi_results.jsonl"
with open(output_file, 'w', encoding='utf-8') as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"Results saved to {output_file}")