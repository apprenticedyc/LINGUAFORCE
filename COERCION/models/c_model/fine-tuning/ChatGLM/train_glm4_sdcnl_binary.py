import pandas as pd
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
train_data_file = "/root/autodl-tmp/DATA/SDCNL/training-set.csv"  # 替换为你的 SDCNL CSV 路径
# 输出目录改名，以防覆盖之前的微调权重
output_dir = "/root/autodl-tmp/fine-tuning/SDCNL/binary/glm4-lora-sdcnl-binary"

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

# ================= 3. 数据集准备 (适配 SDCNL) =================
def load_and_format_sdcnl(file_path):
    formatted_data = []
    print(f"[*] Reading SDCNL data from {file_path}...")
    
    # 读取 CSV，跳过损坏的行，使用 python 引擎增加容错率
    df = pd.read_csv(file_path, on_bad_lines='skip', engine='python')
    
    for _, row in df.iterrows():
        # 1. 提取文本：优先使用 megatext_clean，如果没有则拼接 title 和 selftext
        text = str(row.get("megatext_clean", "")).strip()
        if not text or text.lower() == 'nan':
            title = str(row.get("title", "")).strip()
            selftext = str(row.get("selftext", "")).strip()
            # 如果都没有内容，则跳过
            if (not title or title.lower() == 'nan') and (not selftext or selftext.lower() == 'nan'):
                continue
            text = f"{title}. {selftext}"
            
        # 2. 提取并映射标签 (处理多种可能的格式)
        raw_label = str(row.get("is_suicide", "")).strip().lower()
        
        # 映射为 1 (存在自杀倾向/极端负面情绪)
        if raw_label in ['1', '1.0', 'true', 'suicide']:
            binary_label = 1
        # 映射为 0 (安全/普通抑郁/无自杀倾向)
        elif raw_label in ['0', '0.0', 'false', 'non-suicide', 'non_suicide', 'depression']:
            binary_label = 0
        else:
            continue  # 忽略未知标签

        # 【核心技巧：Prompt 对齐】
        # 把 SDCNL 的长文本伪装成对话的 Utterance 1
        prompt = (
            "Below is a conversation. Please determine whether moral blackmail exists. "
            "Output 1 if it exists; output 0 if it does not.\n"
            "Conversation:\n"
            f"Utterance 1: {text}\n\n"
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

train_dataset = load_and_format_sdcnl(train_data_file)

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
    pin_memory=True,                      # 加速内存传输
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