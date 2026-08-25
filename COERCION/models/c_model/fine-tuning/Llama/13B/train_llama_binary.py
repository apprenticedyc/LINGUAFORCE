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
train_data_path = "/root/autodl-tmp/DATA/train.jsonl"  # 你的训练集路径
output_dir = "/root/autodl-tmp/fine-tuning/binary/Llama13_LoRA/"

# ================= 2. 加载 Tokenizer =================
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

# 【修改】Llama-2 没有默认的 pad_token，显式将 eos_token 设为 pad_token
# 并且为了配合因果语言模型 (CausalLM) 的绝对位置编码，必须设定 padding_side 为 "right"
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# ================= 3. bf16 模型加载 (移除 4-bit 量化) =================
print("[*] Loading Model in bfloat16...")
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    torch_dtype=torch.bfloat16,  # 直接指定以 bfloat16 精度加载，防止默认转为 fp32 导致 OOM
    device_map="auto",
    trust_remote_code=True
)

# 开启梯度检查点，用计算时间换显存空间
model.gradient_checkpointing_enable()
model.enable_input_require_grads() 

# ================= 4. 配置 LoRA 适配器 =================
lora_config = LoraConfig(
    r=16, 
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    # 【注释修正】Llama-2 的线性层名称和 Qwen 一致，保持不变即可覆盖核心注意力层和全连接层
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ================= 5. 数据处理与格式化 =================
def format_prompts(example):
    utterances = example["utterances"]
    non_empty_utts = [utt for utt in utterances if utt.strip() != ""]  
    dialogue_text = "\n".join([f"Utterance{j+1}: {utt}" for j, utt in enumerate(non_empty_utts)])
    label = str(example["dialog_binary_label"])

    system_prompt = "You are a helpful assistant that analyzes conversations."
    prompt_binary = f"Below is a conversation. Please determine whether moral blackmail exists. Output 1 if it exists; output 0 if it does not.\nConversation:\n{dialogue_text}\n\nYour judgment (output only the number 0 or 1):"

    # 【修改】严格按照 Llama-2-Chat 的专属对话模板进行字符串拼接
    # 格式规范: <s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_message} [/INST] {assistant_message} </s>
    # 注意末尾加上 </s> 作为句子结束的标识
    text = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{prompt_binary} [/INST] {label}</s>"
    
    return {"text": text}

dataset = load_dataset("json", data_files={"train": train_data_path})
train_dataset = dataset["train"].map(format_prompts)

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

print("[*] Starting Standard LoRA Fine-tuning with Llama-2...")
trainer.train()

# ================= 8. 保存最终权重 =================
trainer.model.save_pretrained(os.path.join(output_dir, "best_lora"))
tokenizer.save_pretrained(os.path.join(output_dir, "best_lora"))
print("[*] Fine-tuning Complete! LoRA weights saved.")