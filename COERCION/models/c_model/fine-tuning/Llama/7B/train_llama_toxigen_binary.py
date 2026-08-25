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
# 假设你的 ToxiGen 训练集是 CSV 格式，如果是 jsonl 请改回 .jsonl
train_data_path = "/root/autodl-tmp/DATA/ToxiGen/annotated_train.csv"
output_dir = "/root/autodl-tmp/fine-tuning/ToxiGen/binary/Llama7_LoRA_ToxiGen/" # 建议改个名字区分

# ================= 2. 加载 Tokenizer =================
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
# 【修改】Llama-2 没有默认的 pad_token，显式将 eos_token 设为 pad_token
# 并且为了配合因果语言模型 (CausalLM) 的绝对位置编码，必须设定 padding_side 为 "right"
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

# ================= 5. 数据处理与格式化 (针对 ToxiGen 修改) =================
def format_prompts(example):
    # 1. 提取文本并清洗 (处理可能残留的 b'...' 格式)
    raw_text = str(example.get("text", "")).strip()
    if raw_text.startswith("b'") and raw_text.endswith("'"):
        raw_text = raw_text[2:-1]
    elif raw_text.startswith('b"') and raw_text.endswith('"'):
        raw_text = raw_text[2:-1]

    # 2. 标签映射：ToxiGen 的 label 通常是 'hate' 或 'neutral'
    raw_label = str(example.get("label", "")).strip().lower()
    binary_label = "1" if raw_label == "hate" else "0"

    # 3. 构建 System Prompt 和 User Prompt
    system_prompt = "You are a helpful assistant that analyzes text for toxicity and hate speech."
    prompt_binary = f"Below is a text. Please determine whether it contains hate speech or toxicity. Output 1 if it is toxic/hate speech; output 0 if it is neutral/safe.\nText:\n{raw_text}\n\nYour judgment (output only the number 0 or 1):"

    # 【修改】严格按照 Llama-2-Chat 的专属对话模板进行字符串拼接
    # 格式规范: <s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_message} [/INST] {assistant_message}</s>
    text = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{prompt_binary} [/INST] {binary_label}</s>"
    
    return {"text": text}

# 如果你的数据是 CSV 格式，使用 "csv"；如果是 JSONL，改回 "json"
# 注意：直接用 Hugging Face 的 datasets 加载 CSV 比 Pandas 占用内存更小
dataset = load_dataset("csv", data_files={"train": train_data_path})

# 过滤掉文本为空的数据，然后进行 map 映射
train_dataset = dataset["train"].filter(lambda x: x.get("text") is not None and str(x["text"]).strip() != "").map(format_prompts)

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

print("[*] Starting ToxiGen LoRA Fine-tuning with Llama-2...")
trainer.train()

# ================= 8. 保存最终权重 =================
trainer.model.save_pretrained(os.path.join(output_dir, "best_lora"))
tokenizer.save_pretrained(os.path.join(output_dir, "best_lora"))
print("[*] Fine-tuning Complete! LoRA weights saved.")