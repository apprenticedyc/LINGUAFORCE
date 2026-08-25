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
model_name_or_path = "/root/autodl-tmp/Llama-2-13b-chat-ms"
train_data_path = "/root/autodl-tmp/DATA/SemEval-2019_Task5A/train.csv"  # 替换为你的 SemEval 训练集路径
output_dir = "/root/autodl-tmp/fine-tuning/SemEval-2019_Task5A/binary/Llama13_LoRA_SemEval/" # 更新为新的输出目录

# ================= 2. 加载 Tokenizer =================
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
# 【修改】Llama-2 官方词表没有 pad_token，显式将 eos_token 设为 pad_token
# 并且为了配合因果语言模型 (CausalLM) 的训练，设定 padding_side 为 "right"
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# ================= 3. bf16 模型加载 =================
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
    # Llama-2 的线性层名称，保持不变即可覆盖核心注意力层和全连接层
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ================= 5. 数据处理与格式化 =================
def format_semeval_prompts(example):
    # 此处 example 已经是经过清洗的数据字典
    raw_text = example["text"]
    label = example["HS"]

    # 定义 System 和 User 内容
    system_prompt = "You are a helpful assistant that analyzes text for hate speech."
    prompt_binary = f"Below is a text snippet. Please determine whether it contains hate speech. Output 1 if it contains hate speech; output 0 if it does not.\nText:\n{raw_text}\n\nYour judgment (output only the number 0 or 1):"

    # 【修改】使用 Llama-2-Chat 专属指令模板进行拼接
    # 格式: <s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{user} [/INST] {assistant}</s>
    text = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{prompt_binary} [/INST] {label}</s>"
    
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

# 4. 映射为 Llama-2 SFT 所需的格式
train_dataset = hf_dataset.map(format_semeval_prompts)

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
    max_length=512,                # SemEval 单条文本较短，512 完全够用
)

# ================= 7. 启动训练 =================
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

print("[*] Starting Standard LoRA Fine-tuning for SemEval Task 5A with Llama-2...")
trainer.train()

# ================= 8. 保存最终权重 =================
trainer.model.save_pretrained(os.path.join(output_dir, "best_lora"))
tokenizer.save_pretrained(os.path.join(output_dir, "best_lora"))
print("[*] Fine-tuning Complete! LoRA weights saved.")