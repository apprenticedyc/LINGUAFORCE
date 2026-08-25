import json
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

os.environ["OMP_NUM_THREADS"] = "1"

# ================= 1. 配置路径 =================
model_name = "/root/autodl-tmp/glm-4-9b-chat"
train_data_file = "/root/autodl-tmp/DATA/train.jsonl"  # 注意：微调需要使用训练集，不是测试集
output_dir = "/root/autodl-tmp/fine-tuning/ChatGLM/glm4-lora-binary"

# ================= 2. 加载模型和分词器 =================
print("[*] Loading Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

print("[*] Loading Model in bfloat16 for Training...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16, 
    trust_remote_code=True,
    device_map="auto"
)

# 开启梯度检查点以节省显存 (非常关键)
model.gradient_checkpointing_enable()

# ================= 3. 数据集准备 =================
def load_and_format_data(file_path):
    formatted_data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            utterances = item.get("utterances", [])
            true_binary = item.get("dialog_binary_label", -1)
            
            # 过滤异常数据（只取标签为0或1的作为训练语料）
            if true_binary not in [0, 1]:
                continue
                
            non_empty_utts = [utt for utt in utterances if utt.strip() != ""]
            dialogue_text = "\n".join([f"Utterance{j+1}: {utt}" for j, utt in enumerate(non_empty_utts)])
            
            prompt = f"Below is a conversation. Please determine whether moral blackmail exists. Output 1 if it exists; output 0 if it does not.\nConversation:\n{dialogue_text}\n\nYour judgment (output only the number 0 or 1):"
            target = str(true_binary)

            # 构建符合 ChatGLM 格式的多轮对话字典
            formatted_data.append({
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": target}
                ]
            })
    return Dataset.from_list(formatted_data)

print("[*] Loading and formatting dataset...")
train_dataset = load_and_format_data(train_data_file)

# 使用 chat template 将 messages 转换为模型需要的文本格式
def apply_template(example):
    # 将对话格式化为模型所需的 prompt 字符串
    example["text"] = tokenizer.apply_chat_template(
        example["messages"], 
        tokenize=False, 
        add_generation_prompt=False
    )
    return example

train_dataset = train_dataset.map(apply_template)

# ================= 4. LoRA 配置 =================
print("[*] Configuring LoRA...")
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,                     # 秩大小，可调，通常选 8 或 16
    lora_alpha=32,           # 缩放参数，通常是 r 的 2-4 倍
    lora_dropout=0.05,       # 防过拟合
    target_modules=["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"] # GLM4 常见的目标模块
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters() # 打印可训练参数占比（通常 < 1%）

# ================= 5. 训练参数配置 =================
# Data collator: 只计算 assistant 回答部分的 loss，不计算 user prompt 的 loss
response_template = "<|assistant|>" # GLM-4 特定的 assistant token 标记（请根据官方文档微调）
data_collator = DataCollatorForCompletionOnlyLM(response_template=response_template, tokenizer=tokenizer)

training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=4,       # 根据显存大小调整（显存小就设为1或2）
    gradient_accumulation_steps=4,       # 梯度累积，等效于 batch_size = 4 * 4 = 16
    learning_rate=1e-4,                  # LoRA 通常用比全参稍大的学习率
    logging_steps=10,
    num_train_epochs=3,                  # 训练轮数
    save_strategy="epoch",
    bf16=True,                           # 使用 bfloat16 训练
    optim="adamw_torch",
    report_to="none"                     # 如果有 wandb 账号可以改为 "wandb"
)

# ================= 6. 开始训练 =================
print("[*] Starting training...")
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    dataset_text_field="text",
    max_seq_length=1024,                 # 根据你的对话长度调整
    tokenizer=tokenizer,
    args=training_args,
    # data_collator=data_collator          # 如果遇到 collator 报错，可注释掉此行，让模型计算完整文本的 Loss
)

trainer.train()

# ================= 7. 保存 LoRA 权重 =================
print("[*] Saving fine-tuned model...")
trainer.model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print("[*] Done!")