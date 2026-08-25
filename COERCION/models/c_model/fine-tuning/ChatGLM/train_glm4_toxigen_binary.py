import pandas as pd
import json
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig

os.environ["OMP_NUM_THREADS"] = "1"

# ================= 1. 配置路径 =================
model_name = "/root/autodl-tmp/glm-4-9b-chat"
train_data_file = "/root/autodl-tmp/DATA/annotated_train.csv"  # 替换为你的 ToxiGen CSV 路径
# 输出目录改名，以防覆盖之前的微调权重
output_dir = "/root/autodl-tmp/fine-tuning/ToxiGen/binary/glm4-lora-toxigen-binary"

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

# ================= 3. 数据集准备 (适配 ToxiGen) =================
def clean_byte_string(text):
    """去除 CSV 中残留的 b'...' 或 b"..." 格式"""
    text = str(text).strip()
    if text.startswith("b'") and text.endswith("'"):
        return text[2:-1]
    elif text.startswith('b"') and text.endswith('"'):
        return text[2:-1]
    return text

def load_and_format_toxigen(file_path):
    formatted_data = []
    print(f"[*] Reading ToxiGen data from {file_path}...")
    
    # 读取 CSV，跳过损坏的行
    df = pd.read_csv(file_path, on_bad_lines='skip')
    
    for _, row in df.iterrows():
        raw_text = row.get("text", "")
        raw_label = str(row.get("label", "")).strip().lower()
        
        cleaned_text = clean_byte_string(raw_text)
        if not cleaned_text:
            continue
            
        # ToxiGen 标签映射: hate -> 1, neutral -> 0
        if raw_label == 'hate':
            binary_label = 1
        elif raw_label == 'neutral':
            binary_label = 0
        else:
            continue  # 忽略未知标签

        # 【核心技巧：Prompt 对齐】
        # 为了能无缝迁移到你的道德绑架测试集，这里把 ToxiGen 的单句话伪装成对话的 Utterance 1
        # 并使用完全一致的提问方式
        prompt = (
            "Below is a conversation. Please determine whether moral blackmail exists. "
            "Output 1 if it exists; output 0 if it does not.\n"
            "Conversation:\n"
            f"Utterance 1: {cleaned_text}\n\n"
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

train_dataset = load_and_format_toxigen(train_data_file)

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
training_args = SFTConfig(
    output_dir=output_dir,
    per_device_train_batch_size=4,       
    gradient_accumulation_steps=4,       
    learning_rate=1e-4,                  
    logging_steps=10,
    num_train_epochs=3,                  
    save_strategy="epoch",
    bf16=True,                           
    optim="adamw_torch",
    report_to="none",
    dataset_text_field="text",
    max_seq_length=512,
    remove_unused_columns=True
)

# ================= 6. 开始训练 =================
print("[*] Starting transfer learning training...")
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
    args=training_args,
)
# 在 trainer.train() 之前添加：
dataloader = trainer.get_train_dataloader()
trainer.train()

# ================= 7. 保存 LoRA 权重 =================
print("[*] Saving fine-tuned model...")
trainer.model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print("[*] Transfer Learning Training Done!")