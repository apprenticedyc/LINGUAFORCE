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
model_name_or_path = "/root/autodl-tmp/Qwen2.5-14B-Instruct"
train_data_path = "/root/autodl-tmp/DATA/TalkDown/balanced_train.jsonl"  # 替换为你的 TalkDown 训练集路径
output_dir = "/root/autodl-tmp/fine-tuning/TalkDown/binary/Qwen25_LoRA_TalkDown/" # 更新为 TalkDown 的输出目录

# ================= 2. 加载 Tokenizer =================
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
# Qwen2.5 的 pad_token 默认可能未设置，需要补全以防报错
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# ================= 3. bf16 模型加载 =================
print("[*] Loading Model in bfloat16...")
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    torch_dtype=torch.bfloat16,  # 显式指定 bfloat16 防止 OOM
    device_map="auto",
    trust_remote_code=True
)

# 开启梯度检查点，用计算时间换显存空间 (在 32G 显存上微调 14B 必须开启)
model.gradient_checkpointing_enable()
# 注意：开启梯度检查点必须手动加上这行，否则反向传播会报错
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

# ================= 5. 数据处理与格式化 (针对 TalkDown) =================
def filter_invalid_data(example):
    """过滤掉空数据和异常标签"""
    post = str(example.get("post", "")).strip()
    reply = str(example.get("reply", "")).strip()
    raw_label = example.get("label")
    
    # 保证 post 和 reply 至少有一个有内容，且标签是合法的布尔值或其字符串形式
    has_text = bool(post or reply)
    valid_label = raw_label in [True, False, 'true', 'false', 'True', 'False']
    
    return has_text and valid_label

def format_prompts(example):
    """提取原帖和回复，转化为 Prompt，并映射 True/False 标签为 1/0"""
    post = str(example.get("post", "")).strip()
    reply = str(example.get("reply", "")).strip()
    
    # 处理标签映射
    raw_label = example.get("label")
    if raw_label is True or str(raw_label).lower() == 'true':
        label_str = "1"  # 存在居高临下/说教
    else:
        label_str = "0"  # 正常交流

    # 针对 TalkDown 任务设计的 Prompt
    prompt_binary = (
        "Below is a conversation context and a reply. Please determine whether the reply is condescending or patronizing towards the original post. "
        "Output 1 if it is condescending; output 0 if it is normal.\n\n"
        f"Original Post:\n{post}\n\n"
        f"Reply:\n{reply}\n\n"
        "Your judgment (output only the number 0 or 1):"
    )

    messages = [
        {"role": "system", "content": "You are a helpful assistant that analyzes conversations for condescending tone."},
        {"role": "user", "content": prompt_binary},
        {"role": "assistant", "content": label_str} # 训练时，把真实的 0 或 1 作为 assistant 的回答
    ]
    
    # 使用 Qwen 原生的 ChatML 模板渲染文本
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": text}

print("[*] Loading and formatting dataset...")
dataset = load_dataset("json", data_files={"train": train_data_path})
# 先过滤，再映射格式化
train_dataset = dataset["train"].filter(filter_invalid_data).map(format_prompts)
print(f"[*] Dataset ready. Number of training samples: {len(train_dataset)}")

# ================= 6. 训练参数设定 =================
training_args = SFTConfig(
    output_dir=output_dir,
    per_device_train_batch_size=2,  # 如果发生 OOM，可以降为 1，然后增加 gradient_accumulation_steps
    gradient_accumulation_steps=4,  
    learning_rate=2e-4,              
    num_train_epochs=3,              
    logging_steps=10,
    save_strategy="epoch",
    bf16=True,                      
    optim="paged_adamw_8bit",       # 使用 8-bit 优化器节省显存
    max_grad_norm=1.0,
    warmup_steps=100,               
    dataset_text_field="text",      
    max_length=1024,                # TalkDown 的文本通常不会特别长，1024 足够
)

# ================= 7. 启动训练 =================
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

print("[*] Starting Standard LoRA Fine-tuning for TalkDown...")
trainer.train()

# ================= 8. 保存最终权重 =================
trainer.model.save_pretrained(os.path.join(output_dir, "best_lora"))
tokenizer.save_pretrained(os.path.join(output_dir, "best_lora"))
print("[*] Fine-tuning Complete! LoRA weights saved.")