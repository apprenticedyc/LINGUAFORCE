import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

# ================= 1. 基础配置 =================
model_name_or_path = "/root/autodl-tmp/Llama-2-7b-chat-ms"
train_data_path = "/root/autodl-tmp/DATA/SDCNL/training-set.csv"  
output_dir = "/root/autodl-tmp/fine-tuning/SDCNL/binary/Llama7_LoRA_SDCNL/"

# ================= 2. 加载 Tokenizer =================
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
# 【修改】Llama-2 没有默认的 pad_token，必须显式将 eos_token 设为 pad_token
# 并且为了配合 CausalLM 的训练，设定 padding_side 为 "right"
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# ================= 3. bf16 模型加载 =================
print("[*] Loading Model in bfloat16...")
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    torch_dtype=torch.bfloat16,  
    device_map="auto",
    trust_remote_code=True
)

model.gradient_checkpointing_enable()
model.enable_input_require_grads() 

# ================= 4. 配置 LoRA 适配器 =================
lora_config = LoraConfig(
    r=16, 
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    # Llama-2 的线性层名称，保持不变即可覆盖核心注意力层和全连接层
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ================= 5. 数据处理与格式化 (SDCNL Llama-2 适配版) =================
def filter_invalid_data(example):
    """过滤掉没有有效文本或标签异常的脏数据"""
    text = str(example.get("megatext_clean", "")).strip()
    if not text or text.lower() == 'nan':
        title = str(example.get("title", "")).strip()
        selftext = str(example.get("selftext", "")).strip()
        if (not title or title.lower() == 'nan') and (not selftext or selftext.lower() == 'nan'):
            return False 
            
    raw_label = str(example.get("is_suicide", "")).strip().lower()
    valid_labels = ['1', '1.0', 'true', 'suicide', '0', '0.0', 'false', 'non-suicide', 'non_suicide', 'depression']
    if raw_label not in valid_labels:
        return False 
        
    return True

def format_sdcnl_prompts(example):
    """提取文本与标签，并映射为 Llama-2 Chat 模板"""
    text = str(example.get("megatext_clean", "")).strip()
    if not text or text.lower() == 'nan':
        title = str(example.get("title", "")).strip()
        selftext = str(example.get("selftext", "")).strip()
        text = f"{title}. {selftext}"

    raw_label = str(example.get("is_suicide", "")).strip().lower()
    label = "1" if raw_label in ['1', '1.0', 'true', 'suicide'] else "0"

    # 定义 System 和 User 内容
    system_prompt = "You are a highly accurate mental health text analysis assistant."
    prompt_binary = (
        f"Below is a social media post. Please determine whether the author exhibits suicidal ideation "
        f"or extreme negative emotions. Output 1 if they do; output 0 if they do not (e.g., safe, "
        f"general depression, non-suicide).\n\nPost:\n{text}\n\nYour judgment (output only the number 0 or 1):"
    )

    # 【修改】严格按照 Llama-2-Chat 的专属对话模板进行字符串拼接
    formatted_text = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{prompt_binary} [/INST] {label}</s>"
    
    return {"text": formatted_text}

print("[*] Loading SDCNL Dataset...")
dataset = load_dataset("csv", data_files={"train": train_data_path})

print(f"[*] Original dataset size: {len(dataset['train'])}")
train_dataset = dataset["train"].filter(filter_invalid_data)
print(f"[*] Filtered dataset size: {len(train_dataset)}")

# 应用格式化并移除多余的列，防止 Trainer 报错
train_dataset = train_dataset.map(format_sdcnl_prompts, remove_columns=train_dataset.column_names)

# ================= 6. 训练参数设定 =================
training_args = SFTConfig(
    output_dir=output_dir,
    per_device_train_batch_size=2,  
    gradient_accumulation_steps=4,  
    learning_rate=2e-4,             
    num_train_epochs=3,             
    logging_steps=10,
    save_strategy="epoch",
    bf16=True,                      
    optim="paged_adamw_8bit",       
    max_grad_norm=1.0,
    warmup_steps=100,               
    dataset_text_field="text",      
    max_length=1024,                
)

# ================= 7. 启动训练 =================
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

print("[*] Starting SDCNL LoRA Fine-tuning with Llama-2...")
trainer.train()

# ================= 8. 保存最终权重 =================
trainer.model.save_pretrained(os.path.join(output_dir, "best_lora"))
tokenizer.save_pretrained(os.path.join(output_dir, "best_lora"))
print("[*] Fine-tuning Complete! LoRA weights saved.")