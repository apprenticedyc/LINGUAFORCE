import json
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig

# os.environ["OMP_NUM_THREADS"] = "1"

# ================= 1. 配置路径 =================
model_name = "/root/autodl-tmp/glm-4-9b-chat"
train_data_file = "/root/autodl-tmp/DATA/TalkDown/balanced_train.jsonl"  # 替换为你的 TalkDown JSONL 路径
# 输出目录改名，以防覆盖之前的微调权重
output_dir = "/root/autodl-tmp/fine-tuning/TalkDown/binary/glm4-lora-talkdown-binary"

# ================= 2. 加载模型和分词器 =================
print("[*] Loading Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.padding_side = "right"  # 强制右侧填充，防止半精度警告

print("[*] Loading Model in bfloat16 for Training...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16, 
    trust_remote_code=True,
    device_map="cuda"
)

model.gradient_checkpointing_enable()

# ================= 3. 数据集准备 (适配 TalkDown) =================
def load_and_format_talkdown(file_path):
    formatted_data = []
    print(f"[*] Reading TalkDown data from {file_path}...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            
            # 1. 提取完整的对话上下文
            post = str(item.get("post", "")).strip()
            reply = str(item.get("reply", "")).strip()
            
            # 如果内容完全为空则跳过
            if not post and not reply:
                continue
                
            # 2. 提取布尔类型的标签并映射
            raw_label = item.get("label")
            if raw_label is True or str(raw_label).lower() == 'true':
                binary_label = 1  # 存在居高临下/说教倾向
            elif raw_label is False or str(raw_label).lower() == 'false':
                binary_label = 0  # 正常交流
            else:
                continue

            # 【核心技巧：Prompt 对齐】
            # 为了无缝迁移到你的最终测试集，将原帖设为 Utterance 1，回复设为 Utterance 2
            # 提问方式保持与之前的迁移实验完全一致
            prompt = (
                "Below is a conversation. Please determine whether moral blackmail exists. "
                "Output 1 if it exists; output 0 if it does not.\n"
                "Conversation:\n"
                f"Utterance 1: {post}\n"
                f"Utterance 2: {reply}\n\n"
                "Your judgment (output only the number 0 or 1):"
            )
            target = str(binary_label)

            formatted_data.append({
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": target}
                ]
            })
            
    print(f"[*] Successfully loaded {len(formatted_data)} training samples.")
    return Dataset.from_list(formatted_data)

train_dataset = load_and_format_talkdown(train_data_file)

def apply_template(example):
    example["text"] = tokenizer.apply_chat_template(
        example["messages"], 
        tokenize=False, 
        add_generation_prompt=False
    )
    return example

train_dataset = train_dataset.map(apply_template)
# 只保留 text 字段，移除 messages 等
train_dataset = train_dataset.select_columns(["text"])

# ================= 4. LoRA 配置 =================
print("[*] Configuring LoRA...")
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,                     
    lora_alpha=32,           
    lora_dropout=0.05,       
    target_modules=["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"] 
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ================= 5. 训练参数配置 =================
# training_args = SFTConfig(
#     output_dir=output_dir,
#     per_device_train_batch_size=4,       
#     gradient_accumulation_steps=4,       
#     learning_rate=1e-4,                  
#     logging_steps=10,
#     num_train_epochs=3,                  
#     save_strategy="epoch",
#     bf16=True,                           
#     optim="adamw_torch",
#     report_to="none",
#     dataset_text_field="text",
#     max_seq_length=512,
#     remove_unused_columns=True
# )
training_args = SFTConfig(
    output_dir=output_dir,
    per_device_train_batch_size=4,       # 若显存允许可提高至 8
    gradient_accumulation_steps=4,       # 保持有效 batch size 16
    learning_rate=1e-4,
    logging_steps=10,
    num_train_epochs=3,
    save_strategy="epoch",
    bf16=True,
    optim="adamw_torch",
    report_to="none",
    dataset_text_field="text",
    max_seq_length=512,
    remove_unused_columns=True,
    dataloader_num_workers=4,            # 启用多进程数据加载
    # pin_memory=True,                      # 加速内存传输
)

# ================= 6. 开始训练 =================
print("[*] Starting transfer learning training...")
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
    args=training_args,
)

dataloader = trainer.get_train_dataloader()
trainer.train()

# ================= 7. 保存 LoRA 权重 =================
print("[*] Saving fine-tuned model...")
trainer.model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print("[*] Transfer Learning Training Done!")