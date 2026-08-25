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
train_data_path = "/root/autodl-tmp/DATA/train.jsonl"  # 你的训练集路径
output_dir = "/root/autodl-tmp/fine-tuning/multi/Qwen25_LoRA/"

# ================= 2. 加载 Tokenizer =================
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
# Qwen2.5 的 pad_token 默认可能未设置，需要补全以防报错
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# ================= 3. bf16 模型加载 (移除 4-bit 量化) =================
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

# ================= 5. 数据处理与格式化 =================
def format_prompts(example):
    utterances = example["utterances"]
    non_empty_utts = [utt for utt in utterances if utt.strip() != ""]  # 使用strip()可同时过滤全空格的情况
    dialogue_text = "\n".join([f"Utterance{j+1}: {utt}" for j, utt in enumerate(non_empty_utts)])
    label = str(example["dialog_multi_label"])

    prompt_multi = (
        "Below is a conversation. Please assess the severity of moral blackmail "
        "(0 indicates no blackmail, 5 indicates extreme blackmail).\n"
        "0: No blackmail, 1: Slight suspicion, 2: Slight tendency towards moral blackmail, "
        "3: Clear implication, 4: Highly self-centered, 5: Extreme blackmail.\n"
        f"Conversation:\n{dialogue_text}\n\n"
        "Your judgment (output only a number between 0 and 5):"
    )

    messages = [
        {"role": "system", "content": "You are a helpful assistant that analyzes conversations."},
        {"role": "user", "content": prompt_multi},
        {"role": "assistant", "content": label} # 微调时，直接将答案塞入 assistant
    ]
    
    # 使用 Qwen 原生的 ChatML 模板渲染文本
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": text}

dataset = load_dataset("json", data_files={"train": train_data_path})
train_dataset = dataset["train"].map(format_prompts)

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
    optim="paged_adamw_8bit",       # 继续保留 8-bit 优化器，能省下不少显存
    max_grad_norm=1.0,
    warmup_steps=100,               
    dataset_text_field="text",      
    max_length=1024,                # 如果 OOM 且你的对话普遍较短，可以把这里下调到 1024 或 512
)

# ================= 7. 启动训练 =================
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

print("[*] Starting Standard LoRA Fine-tuning...")
trainer.train()

# ================= 8. 保存最终权重 =================
trainer.model.save_pretrained(os.path.join(output_dir, "best_lora"))
tokenizer.save_pretrained(os.path.join(output_dir, "best_lora"))
print("[*] Fine-tuning Complete! LoRA weights saved.")