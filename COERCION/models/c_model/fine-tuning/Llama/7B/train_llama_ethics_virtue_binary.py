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
# 【确认】这里已经是 Llama-2 的路径
model_name_or_path = "/root/autodl-tmp/Llama-2-7b-chat-ms"
train_data_path = "/root/autodl-tmp/DATA/Ethics/virtue/virtue_train.csv"  
output_dir = "/root/autodl-tmp/fine-tuning/Ethics/virtue/binary/Llama7_LoRA_Virtue/"

# ================= 2. 加载 Tokenizer =================
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

# 【修改】Llama-2 默认没有 pad_token，通常将 eos_token 设为 pad_token
# 并且为了让 CausalLM 训练更稳定，显式设置 padding_side 为 "right"
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
    # 【修改注释】Llama-2 的线性层名称和 Qwen 是一致的，保持不变即可覆盖核心权重
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ================= 5. 数据处理与格式化 =================
def format_prompts(example):
    scenario = str(example["scenario"]).strip()
    label = str(example["label"]).strip()

    # 定义 System 和 User 内容
    system_prompt = "You are a helpful assistant that assesses the moral virtue of various scenarios."
    user_prompt = f"Below is a scenario. Please evaluate the moral virtue of the action or situation. Output 1 if it is virtuous or morally acceptable; output 0 if it is not.\nScenario:\n{scenario}\n\nYour judgment (output only the number 0 or 1):"

    # 【关键修改】严格按照 Llama-2-Chat 的对话模板进行字符串拼接
    # 格式规范: <s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_message} [/INST] {assistant_message} </s>
    # 注意末尾加上 </s> 作为句子结束的标识
    text = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_prompt} [/INST] {label}</s>"
    
    return {"text": text}

dataset = load_dataset("csv", data_files={"train": train_data_path})

# 清洗数据：过滤掉 scenario 为空或为 nan 的无效行
dataset = dataset.filter(lambda x: x["scenario"] is not None and str(x["scenario"]).strip() != "" and str(x["scenario"]).lower() != "nan")

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
    max_length=256,                 
)

# ================= 7. 启动训练 =================
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

print("[*] Starting Standard LoRA Fine-tuning on Virtue Dataset using Llama-2...")
trainer.train()

# ================= 8. 保存最终权重 =================
trainer.model.save_pretrained(os.path.join(output_dir, "best_lora"))
tokenizer.save_pretrained(os.path.join(output_dir, "best_lora"))
print("[*] Fine-tuning Complete! LoRA weights saved.")