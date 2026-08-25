import os
import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

# ================= 1. 基础配置 =================
model_name_or_path = "/root/autodl-tmp/Qwen2.5-14B-Instruct"
train_data_path = "/root/autodl-tmp/DATA/SemEval-2019_Task5A/train.csv"  # 替换为你的 SemEval 训练集路径
output_dir = "/root/autodl-tmp/fine-tuning/SemEval-2019_Task5A/binary/Qwen25_LoRA_SemEval/" # 更新为新的输出目录

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
def format_semeval_prompts(example):
    # 此处 example 已经是经过清洗的数据字典
    raw_text = example["text"]
    label = example["HS"]

    # 针对仇恨言论（Hate Speech）重写 Prompt
    prompt_binary = f"Below is a text snippet. Please determine whether it contains hate speech. Output 1 if it contains hate speech; output 0 if it does not.\nText:\n{raw_text}\n\nYour judgment (output only the number 0 or 1):"

    messages = [
        {"role": "system", "content": "You are a helpful assistant that analyzes text for hate speech."},
        {"role": "user", "content": prompt_binary},
        {"role": "assistant", "content": label} # 微调时，直接将答案塞入 assistant
    ]
    
    # 使用 Qwen 原生的 ChatML 模板渲染文本
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": text}

print("[*] Loading and cleaning SemEval dataset...")
# 1. 使用 Pandas 强健地读取 CSV，跳过损坏的行
df_train = pd.read_csv(train_data_path, on_bad_lines='skip', engine='python')

# 2. 数据清洗：过滤掉没有文本的行，并规范化 HS 标签
valid_rows = []
for _, row in df_train.iterrows():
    text = str(row.get("text", "")).strip()
    raw_hs = str(row.get("HS", "")).strip()
    
    if not text or text.lower() == 'nan':
        continue
        
    if raw_hs in ['1', '1.0', 'true']:
        binary_label = '1'
    elif raw_hs in ['0', '0.0', 'false']:
        binary_label = '0'
    else:
        continue # 忽略异常标签
        
    valid_rows.append({"text": text, "HS": binary_label})

# 3. 转换为 HuggingFace Dataset
cleaned_df = pd.DataFrame(valid_rows)
print(f"[*] Usable training samples after cleaning: {len(cleaned_df)}")
hf_dataset = Dataset.from_pandas(cleaned_df)

# 4. 映射为 Qwen SFT 所需的 ChatML 格式
train_dataset = hf_dataset.map(format_semeval_prompts)

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
    max_length=512,                # SemEval 单条文本较短，1024 完全够用，甚至可降到 512
)

# ================= 7. 启动训练 =================
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

print("[*] Starting Standard LoRA Fine-tuning for SemEval Task 5A...")
trainer.train()

# ================= 8. 保存最终权重 =================
trainer.model.save_pretrained(os.path.join(output_dir, "best_lora"))
tokenizer.save_pretrained(os.path.join(output_dir, "best_lora"))
print("[*] Fine-tuning Complete! LoRA weights saved.")