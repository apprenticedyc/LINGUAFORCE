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
train_data_file = "/root/autodl-tmp/DATA/train_multi.jsonl"  # 确保换成多分类的数据集路径
output_dir = "/root/autodl-tmp/fine-tuning/ChatGLM/glm4-lora-multiclass"

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

# 开启梯度检查点以节省显存
model.gradient_checkpointing_enable()

# ================= 3. 数据集准备 (多分类修改核心) =================
def load_and_format_data(file_path):
    formatted_data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            utterances = item.get("utterances", [])
            
            # 【注意修改点 1】：你需要确保 JSONL 中的标签字段名是正确的。
            # 假设你的多分类标签字段名是 "dialog_severity_label"
            true_label = item.get("dialog_severity_label", -1) 
            
            # 【注意修改点 2】：过滤异常数据，放行 0 到 5 的标签
            if true_label not in [0, 1, 2, 3, 4, 5]:
                continue
                
            non_empty_utts = [utt for utt in utterances if utt.strip() != ""]
            dialogue_text = "\n".join([f"Utterance{j+1}: {utt}" for j, utt in enumerate(non_empty_utts)])
            
            # 【注意修改点 3】：应用你提供的多分类 Prompt
            prompt = (
                "You are an expert in moral coercion detection. Please read the following dialogue to assess the severity of your moral coercion and strictly output a numerical label between 0 and 5.\n"
                "Classification criteria:\n"
                "0: No coercion (normal communication) \n"
                "1: Minor suspicion (ambiguous)\n"
                "2: Slight bias (with slight pressure)\n"
                "3: Obvious suggestion (obvious moral pressure)\n"
                "4: Highly self-centered (ignoring others' feelings and forcefully exerting pressure)\n"
                "5: Extreme coercion (Strong Moral coercion and Accusation)\n\n"
                f"Dialogue Content: \n{dialogue_text}\n\n"
                "Your judgment (please only output a pure number between 0-5, do not explain):"
            )
            target = str(true_label)

            formatted_data.append({
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": target}
                ]
            })
    return Dataset.from_list(formatted_data)

print("[*] Loading and formatting dataset...")
train_dataset = load_and_format_data(train_data_file)

def apply_template(example):
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
    r=8,                     
    lora_alpha=32,           
    lora_dropout=0.05,       
    target_modules=["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"] 
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters() 

# ================= 5. 训练参数配置 =================
response_template = "<|assistant|>" 
data_collator = DataCollatorForCompletionOnlyLM(response_template=response_template, tokenizer=tokenizer)

training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=4,       
    gradient_accumulation_steps=4,       
    learning_rate=1e-4,                  
    logging_steps=10,
    num_train_epochs=3,                  
    save_strategy="epoch",
    bf16=True,                           
    optim="adamw_torch",
    report_to="none"                     
)

# ================= 6. 开始训练 =================
print("[*] Starting training...")
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    dataset_text_field="text",
    max_seq_length=1024,                 
    tokenizer=tokenizer,
    args=training_args,
    data_collator=data_collator          # 如果运行报错，可以尝试注释掉这一行
)

trainer.train()

# ================= 7. 保存 LoRA 权重 =================
print("[*] Saving fine-tuned model...")
trainer.model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print("[*] Done!")