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
model_name_or_path = "/root/autodl-tmp/Llama-2-13b-chat-ms"
train_data_path = "/root/autodl-tmp/DATA/TalkDown/balanced_train.jsonl"  # 替换为你的 TalkDown 训练集路径
output_dir = "/root/autodl-tmp/fine-tuning/TalkDown/binary/Llama13_LoRA_TalkDown/" # 更新为 TalkDown 的输出目录

# ================= 2. 加载 Tokenizer =================
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
# 【修改】Llama-2 没有默认的 pad_token，显式将 eos_token 设为 pad_token
# 并且为了配合因果语言模型 (CausalLM) 的训练，必须设定 padding_side 为 "right"
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# ================= 3. bf16 模型加载 =================
print("[*] Loading Model in bfloat16...")
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    torch_dtype=torch.bfloat16,  # 显式指定 bfloat16 防止 OOM
    device_map="auto",
    trust_remote_code=True
)

# 开启梯度检查点，用计算时间换显存空间 (在 32G 显存上微调 14B/7B 建议开启)
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

    # 定义 System 和 User 内容
    system_prompt = "You are a helpful assistant that analyzes conversations for condescending tone."
    prompt_binary = (
        "Below is a conversation context and a reply. Please determine whether the reply is condescending or patronizing towards the original post. "
        "Output 1 if it is condescending; output 0 if it is normal.\n\n"
        f"Original Post:\n{post}\n\n"
        f"Reply:\n{reply}\n\n"
        "Your judgment (output only the number 0 or 1):"
    )

    # 【修改】使用 Llama-2-Chat 专属指令模板进行拼接
    # 格式: <s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{user} [/INST] {assistant}</s>
    text = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{prompt_binary} [/INST] {label_str}</s>"
    
    return {"text": text}

print("[*] Loading and formatting dataset...")
dataset = load_dataset("json", data_files={"train": train_data_path})
# 先过滤，再映射格式化
train_dataset = dataset["train"].filter(filter_invalid_data).map(format_prompts)
print(f"[*] Dataset ready. Number of training samples: {len(train_dataset)}")

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
    max_length=1024,                # TalkDown 的文本通常不会特别长，1024 足够
)

# ================= 7. 启动训练 =================
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

print("[*] Starting Standard LoRA Fine-tuning for TalkDown with Llama-2...")
trainer.train()

# ================= 8. 保存最终权重 =================
trainer.model.save_pretrained(os.path.join(output_dir, "best_lora"))
tokenizer.save_pretrained(os.path.join(output_dir, "best_lora"))
print("[*] Fine-tuning Complete! LoRA weights saved.")