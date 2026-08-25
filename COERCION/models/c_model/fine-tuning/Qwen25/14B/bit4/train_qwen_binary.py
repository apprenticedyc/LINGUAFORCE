import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# ================= 1. 基础配置 =================
model_name_or_path = "/root/autodl-tmp/Qwen2.5-14B-Instruct"
train_data_path = "/root/autodl-tmp/DATA/train.jsonl"  # 你的训练集路径
output_dir = "/root/autodl-tmp/fine-tuning/binary/Qwen25_LoRA/"

# ================= 2. 加载 Tokenizer =================
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
# Qwen2.5 的 pad_token 默认可能未设置，需要补全以防报错
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# ================= 3. 4-bit 量化与模型加载 =================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16 # 建议使用 bfloat16 防溢出
)

print("[*] Loading Model in 4-bit...")
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

# 开启梯度检查点，用计算时间换显存空间
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

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
    # dialogue_text = "\n".join([f"第{j+1}句：{utt}" for j, utt in enumerate(utterances)])
    non_empty_utts = [utt for utt in utterances if utt.strip() != ""]  # 使用strip()可同时过滤全空格的情况
    dialogue_text = "\n".join([f"Utterance{j+1}: {utt}" for j, utt in enumerate(non_empty_utts)])
    label = str(example["dialog_binary_label"])

    prompt_binary = f"Below is a conversation. Please determine whether moral blackmail exists. Output 1 if it exists; output 0 if it does not.\nConversation:\n{dialogue_text}\n\nYour judgment (output only the number 0 or 1):"

    messages = [
        {"role": "system", "content": "You are a helpful assistant that analyzes conversations."},
        {"role": "user", "content": prompt_binary},
        {"role": "assistant", "content": label} # 微调时，直接将答案塞入 assistant
    ]
    
    # 使用 Qwen 原生的 ChatML 模板渲染文本
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": text}

dataset = load_dataset("json", data_files={"train": train_data_path})
train_dataset = dataset["train"].map(format_prompts)

# DataCollator 核心逻辑：只对模型生成的数字（0-5）计算 Loss，忽略 Prompt 部分的 Loss
# Qwen 的 assistant 起始标识符是 <|im_start|>assistant\n
# response_template = "<|im_start|>assistant\n"
# collator = DataCollatorForCompletionOnlyLM(response_template=response_template, tokenizer=tokenizer)

# ================= 6. 训练参数设定 (使用 SFTConfig) =================
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
    warmup_steps=100,               # <--- 替换掉 deprecated 的 warmup_ratio
    dataset_text_field="text",      # <--- 移到这里
    max_length=1536,            # <--- 移到这里
)

# ================= 7. 启动训练 =================
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    # 这里不再需要 dataset_text_field 和 max_seq_length
)

print("[*] Starting QLoRA Fine-tuning...")
trainer.train()

# ================= 8. 保存最终权重 =================
trainer.model.save_pretrained(os.path.join(output_dir, "best_lora"))
tokenizer.save_pretrained(os.path.join(output_dir, "best_lora"))
print("[*] Fine-tuning Complete! LoRA weights saved.")