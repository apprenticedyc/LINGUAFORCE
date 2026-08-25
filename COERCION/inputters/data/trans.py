import json

def jsonl_to_label_studio_json(input_filepath, output_filepath):
    label_studio_data = []
    
    with open(input_filepath, 'r', encoding='utf-8') as infile:
        for line in infile:
            if not line.strip():
                continue
                
            data = json.loads(line)
            utterances = data.get('utterances', [])
            
            # 构造对话列表，交替分配 Person1 和 Person2
            dialogue_list = []
            for i, utterance in enumerate(utterances):
                author = "Person1" if i % 2 == 0 else "Person2"
                dialogue_list.append({
                    "author": author,
                    "text": utterance
                })
            
            # 构建单条任务数据
            task_data = {
                "dialogue": dialogue_list,
                "dialogue_id": data.get("dialogue_id"),
                "dialog_multi_label": data.get("dialog_multi_label"),
                "dialog_binary_label": data.get("dialog_binary_label")
            }
            label_studio_data.append(task_data)

    # 写入为标准 JSON 数组文件
    with open(output_filepath, 'w', encoding='utf-8') as outfile:
        json.dump(label_studio_data, outfile, ensure_ascii=False, indent=2)

# 运行脚本
jsonl_to_label_studio_json('inputters/data/data.jsonl', 'inputters/data/origin/data.json')
print("转换完成！可以直接导入 Label Studio 了。")