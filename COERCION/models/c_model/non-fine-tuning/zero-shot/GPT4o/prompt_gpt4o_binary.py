import json

# ================= 1. 配置路径 =================
input_file = "GPT4o/zero-shot/data/test.jsonl"
output_file = "GPT4o/zero-shot/prompt/prompt_test.jsonl"

# ================= 2. 定义 Prompt 模板 =================
# 将第一个代码中的 prompt_binary 提取出来作为静态模板
# 留出 {dialogue_text} 作为后续填充对话内容的占位符
prompt_template = """Below is a conversation. Please determine whether moral blackmail exists. Output 1 if it exists; output 0 if it does not.
Conversation:
{dialogue_text}

Your judgment (output only the number 0 or 1):"""

# ================= 3. 数据处理与写入 =================
processed_count = 0

with open(input_file, "r", encoding="utf-8") as fin, open(output_file, "w", encoding="utf-8") as fout:
    for idx, line in enumerate(fin):
        if not line.strip():  # 跳过可能的空行
            continue
            
        dialogue = json.loads(line.strip())
        
        # 构建对话文本 (结合了空字符串过滤和 Person1/Person2 角色交替)
        dialogue_text_lines = []
        valid_turn_count = 0  # 独立计数器，确保 Person1 和 Person2 绝对交替，不受空字符串影响
        
        for i, utt in enumerate(dialogue.get("utterances", [])):
            utt_stripped = utt.strip()
            if len(utt_stripped) > 0:
                # # 采用 Person1: 和 Person2: 的格式，通常比纯粹的 Utterance 1, 2 更利于 LLM 理解对话语境
                # speaker = "Person1" if valid_turn_count % 2 == 0 else "Person2"
                # dialogue_text_lines.append(f"{speaker}: {utt_stripped}")
                # valid_turn_count += 1
                
                dialogue_text_lines.append(f"Utterance{i+1}: {utt_stripped}")
        
        # 将列表拼接成一段完整的字符串，用换行符隔开
        dialogue_text = "\n".join(dialogue_text_lines)
        
        # 将拼接好的对话注入到 Prompt 模板中
        prompt_text = prompt_template.format(dialogue_text=dialogue_text)
        
        # 组装输出的字典结构
        output_dict = {
            # 使用 .get() 并提供默认值，防止某些行缺失 dialogue_id 导致报错
            "dialogue_id": dialogue.get("dialogue_id", f"missing_id_{idx}"), 
            "prompt": prompt_text
        }
        
        # 写入新的 JSONL 文件
        fout.write(json.dumps(output_dict, ensure_ascii=False) + "\n")
        processed_count += 1

print(f"[*] 处理完毕！共生成 {processed_count} 条 Prompt。")
print(f"[*] 文件已保存至: {output_file}")