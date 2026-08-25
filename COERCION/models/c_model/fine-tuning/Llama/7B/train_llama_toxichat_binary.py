import os
import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

# ================= 1. 基础配置 =================
model_name_or_path = "/root/autodl-tmp/Llama-2-7b-chat-ms"
# 将这里的路径换成你的 ToxiChat jsonl 文件路径
train_data_path = "/root/autodl-tmp/DATA/ToxiChat/train.jsonl"  
output_dir = "/root/autodl-tmp/fine-tuning/ToxiChat/binary/Llama7_LoRA_ToxiChat/"

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

# ================= 5. 数据处理与格式化 (针对 ToxiChat Llama-2 适配) =================
def load_and_format_toxichat(file_path, tokenizer):
    """
    读取 ToxiChat jsonl 并直接转化为适用于 SFTTrainer 的 HuggingFace Dataset
    """
    formatted_data = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            item = json.loads(line)
            
            # 1. 提取上下文 Thread
            thread = item.get("reddit_thread", [])
            context_utterances = [turn["text"].strip() for turn in thread]
            # 拼接历史上下文，标明轮次
            context_text = "\n".join([f"Turn {i+1}: {text}" for i, text in enumerate(context_utterances)])
            
            # 2. 定义处理单个回复的内部函数
            def _create_sample(response_dict):
                response_text = response_dict.get("text", "").strip()
                # 标签转换：Safe -> 0, Offensive/Toxic -> 1
                offense_label = response_dict.get("offense_label", "Safe")
                label = "0" if offense_label.lower() == "safe" else "1"
                
                system_prompt = "You are a helpful assistant that detects toxic and offensive language."
                prompt = (
                    "Given the conversation context, determine if the following reply is toxic, offensive, or unsafe. "
                    "Output 1 if it is toxic/offensive; output 0 if it is safe.\n\n"
                    f"Context:\n{context_text}\n\n"
                    f"Reply:\n{response_text}\n\n"
                    "Your judgment (output only the number 0 or 1):"
                )
                
                # 【修改】严格按照 Llama-2-Chat 的专属对话模板进行字符串拼接
                # 格式规范: <s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_message} [/INST] {assistant_message}</s>
                text = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{prompt} [/INST] {label}</s>"
                return {"text": text}
            
            # 3. 提取 dgpt 和 gpt3 的回复，分别构建样本 (展平操作)
            if "final_dgpt_response" in item and item["final_dgpt_response"]:
                formatted_data.append(_create_sample(item["final_dgpt_response"]))
                
            if "final_gpt3_response" in item and item["final_gpt3_response"]:
                formatted_data.append(_create_sample(item["final_gpt3_response"]))

    # 将 Python List 直接转换为 Hugging Face 的 Dataset 对象
    return Dataset.from_list(formatted_data)

print("[*] Processing ToxiChat Dataset...")
train_dataset = load_and_format_toxichat(train_data_path, tokenizer)
print(f"[*] Total training samples after flattening: {len(train_dataset)}")

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

print("[*] Starting Transfer Learning on ToxiChat with Llama-2...")
trainer.train()

# ================= 8. 保存最终权重 =================
trainer.model.save_pretrained(os.path.join(output_dir, "best_lora"))
tokenizer.save_pretrained(os.path.join(output_dir, "best_lora"))
print("[*] Fine-tuning Complete! LoRA weights saved.")