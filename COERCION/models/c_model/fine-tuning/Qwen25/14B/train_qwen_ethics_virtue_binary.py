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
# 【修改】指向你的 Virtue 训练集 CSV 文件
train_data_path = "/root/autodl-tmp/DATA/Ethics/virtue/virtue_train.csv"  
# 【修改】更新输出目录名，避免覆盖之前的实验
output_dir = "/root/autodl-tmp/fine-tuning/Ethics/virtue/binary/Qwen25_LoRA_Virtue/"

# ================= 2. 加载 Tokenizer =================
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
# Qwen2.5 的 pad_token 默认可能未设置，需要补全以防报错
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# ================= 3. bf16 模型加载 =================
print("[*] Loading Model in bfloat16...")
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    torch_dtype=torch.bfloat16,  # 直接指定以 bfloat16 精度加载
    device_map="auto",
    trust_remote_code=True
)

# 开启梯度检查点，用计算时间换显存空间 (在 32G 显存上微调 14B 必须开启)
model.gradient_checkpointing_enable()
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
    # 提取 Virtue 的情境和标签
    scenario = str(example["scenario"]).strip()
    label = str(example["label"]).strip()

    # 针对美德/道德评估设计全新的 Prompt
    prompt_binary = f"Below is a scenario. Please evaluate the moral virtue of the action or situation. Output 1 if it is virtuous or morally acceptable; output 0 if it is not.\nScenario:\n{scenario}\n\nYour judgment (output only the number 0 or 1):"

    messages = [
        {"role": "system", "content": "You are a helpful assistant that assesses the moral virtue of various scenarios."},
        {"role": "user", "content": prompt_binary},
        {"role": "assistant", "content": label} # 微调时，直接将答案塞入 assistant
    ]
    
    # 使用 Qwen 原生的 ChatML 模板渲染文本
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": text}

# 【修改】使用 csv 模式加载数据
dataset = load_dataset("csv", data_files={"train": train_data_path})

# 【新增】清洗数据：过滤掉 scenario 为空或为 nan 的无效行
dataset = dataset.filter(lambda x: x["scenario"] is not None and str(x["scenario"]).strip() != "" and str(x["scenario"]).lower() != "nan")

train_dataset = dataset["train"].map(format_prompts)

# ================= 6. 训练参数设定 =================
training_args = SFTConfig(
    output_dir=output_dir,
    per_device_train_batch_size=2,  # 显存充足的话可以尝试调高至 4 或 8
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
    # 【修改】Virtue 文本较短，将最大长度下调至 256 以加快训练速度并降低显存占用
    max_length=256,                 
)

# ================= 7. 启动训练 =================
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

print("[*] Starting Standard LoRA Fine-tuning on Virtue Dataset...")
trainer.train()

# ================= 8. 保存最终权重 =================
trainer.model.save_pretrained(os.path.join(output_dir, "best_lora"))
tokenizer.save_pretrained(os.path.join(output_dir, "best_lora"))
print("[*] Fine-tuning Complete! LoRA weights saved.")