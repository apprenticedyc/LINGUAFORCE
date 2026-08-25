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
# 修改为你的 Implicit Hate CSV 训练集路径
train_data_path = "/root/autodl-tmp/DATA/Implicit_Hate/train.csv"  
output_dir = "/root/autodl-tmp/fine-tuning/Implicit_Hate/binary/Llama7_LoRA_ImplicitHate/"

# ================= 2. 加载 Tokenizer =================
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
# 【修改】Llama-2 默认没有 pad_token，通常将 eos_token 设为 pad_token
# 并且为了让 CausalLM 训练更稳定，必须显式设置 padding_side 为 "right"
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

# ================= 5. 数据处理与格式化 (核心修改区) =================
def format_prompts(example):
    # 提取 post 和 class，并处理潜在的空值
    raw_text = str(example.get("post", "")).strip()
    raw_label = str(example.get("class", "")).strip().lower()
    
    # 标签映射：not_hate -> 0, implicit_hate/explicit_hate -> 1
    if raw_label == 'not_hate':
        label = "0"
    elif raw_label in ['implicit_hate', 'explicit_hate']:
        label = "1"
    else:
        label = "-1" # 标记异常标签，后续过滤掉

    # 定义 System 和 User 内容
    system_prompt = "You are a helpful assistant that analyzes text for hate speech."
    user_prompt = f"Below is a social media post. Please determine whether it contains hate speech (either implicit or explicit). Output 1 if it contains hate speech; output 0 if it does not.\nPost:\n{raw_text}\n\nYour judgment (output only the number 0 or 1):"

    # 【修改】严格按照 Llama-2-Chat 的专属对话模板进行字符串拼接
    # 格式规范: <s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_message} [/INST] {assistant_message} </s>
    text = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_prompt} [/INST] {label}</s>"
    
    # 增加一个 valid 字段，用于后续过滤无效数据
    is_valid = (label != "-1") and (raw_text != "") and (raw_text.lower() != "nan")
    return {"text": text, "valid": is_valid}

# 改用 "csv" 加载数据集
dataset = load_dataset("csv", data_files={"train": train_data_path})

# 映射 Prompt 并过滤掉异常或空数据
train_dataset = dataset["train"].map(format_prompts)
train_dataset = train_dataset.filter(lambda x: x["valid"])

print(f"[*] Valid training samples after filtering: {len(train_dataset)}")

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
    max_length=256,                 # 社交媒体 post 通常较短，下调到 256 可以节省大量显存并加快训练
)

# ================= 7. 启动训练 =================
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

print("[*] Starting Transfer Learning LoRA Fine-tuning on Implicit Hate using Llama-2...")
trainer.train()

# ================= 8. 保存最终权重 =================
trainer.model.save_pretrained(os.path.join(output_dir, "best_lora"))
tokenizer.save_pretrained(os.path.join(output_dir, "best_lora"))
print("[*] Fine-tuning Complete! LoRA weights saved.")