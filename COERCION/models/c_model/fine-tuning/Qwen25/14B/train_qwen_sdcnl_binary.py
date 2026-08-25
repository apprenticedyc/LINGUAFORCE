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
model_name_or_path = "/root/autodl-tmp/Qwen25-14B-Instruct"
train_data_path = "/root/autodl-tmp/DATA/SDCNL/training-set.csv"  # 修改为了 SDCNL 的训练集 CSV 路径
output_dir = "/root/autodl-tmp/fine-tuning/SDCNL/binary/Qwen25_LoRA_SDCNL/"

# ================= 2. 加载 Tokenizer =================
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
# Qwen2.5 的 pad_token 默认可能未设置，需要补全以防报错
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# ================= 3. bf16 模型加载 =================
print("[*] Loading Model in bfloat16...")
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    torch_dtype=torch.bfloat16,  # 直接指定以 bfloat16 精度加载，防止默认转为 fp32 导致 OOM
    device_map="auto",
    trust_remote_code=True
)

# 开启梯度检查点，用计算时间换显存空间 (在 32G 显存上微调 14B 必须开启)
model.gradient_checkpointing_enable()
# 注意：移除了 kbit training 准备函数后，开启梯度检查点必须手动加上这行，否则反向传播会报错
model.enable_input_require_grads() 

# ================= 4. 配置 LoRA 适配器 =================
lora_config = LoraConfig(
    r=16, 
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    # 针对 Qwen2.5 架构的核心注意力层和全连接层
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ================= 5. 数据处理与格式化 (SDCNL 适配版) =================
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
    """提取文本与标签，并映射为 Qwen ChatML 模板"""
    text = str(example.get("megatext_clean", "")).strip()
    if not text or text.lower() == 'nan':
        title = str(example.get("title", "")).strip()
        selftext = str(example.get("selftext", "")).strip()
        text = f"{title}. {selftext}"

    raw_label = str(example.get("is_suicide", "")).strip().lower()
    label = "1" if raw_label in ['1', '1.0', 'true', 'suicide'] else "0"

    prompt_binary = (
        f"Below is a social media post. Please determine whether the author exhibits suicidal ideation "
        f"or extreme negative emotions. Output 1 if they do; output 0 if they do not (e.g., safe, "
        f"general depression, non-suicide).\n\nPost:\n{text}\n\nYour judgment (output only the number 0 or 1):"
    )

    messages = [
        {"role": "system", "content": "You are a highly accurate mental health text analysis assistant."},
        {"role": "user", "content": prompt_binary},
        {"role": "assistant", "content": label} 
    ]
    
    formatted_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
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
    per_device_train_batch_size=2,  # 如果发生 OOM，可以将这里降为 1，然后增加下面 accum 的值
    gradient_accumulation_steps=4,  
    learning_rate=2e-4,             
    num_train_epochs=3,             
    logging_steps=10,
    save_strategy="epoch",
    bf16=True,                      
    optim="paged_adamw_8bit",       # 保留 8-bit 优化器节省显存
    max_grad_norm=1.0,
    warmup_steps=100,               
    dataset_text_field="text",      
    max_length=1024,                # 社交媒体帖子通常较长，1024是个安全的截断长度
)

# ================= 7. 启动训练 =================
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

print("[*] Starting SDCNL LoRA Fine-tuning...")
trainer.train()

# ================= 8. 保存最终权重 =================
trainer.model.save_pretrained(os.path.join(output_dir, "best_lora"))
tokenizer.save_pretrained(os.path.join(output_dir, "best_lora"))
print("[*] Fine-tuning Complete! LoRA weights saved.")