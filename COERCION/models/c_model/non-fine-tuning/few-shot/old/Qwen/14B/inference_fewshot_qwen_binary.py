import json
import random
from transformers import AutoModelForCausalLM, AutoTokenizer

# ================= 配置区域 =================
model_name = "/root/autodl-tmp/Qwen2.5-14B-Instruct"
data_file = "/root/autodl-tmp/DATA/test.jsonl"
output_file = "/root/autodl-tmp/non-fine-tuning/Qwen25/few_shot_results.jsonl"
max_samples = 99999
# ===========================================

# 1. 加载模型
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")

# 2. 定义你的 Few-Shot 样本池 (来源于你提供的数据)
# label=0: 无道德绑架, label=1: 有道德绑架
example_pool = [
    # --- 负样本 (Label 0) ---
    {"id": 3764, "utts": ["You should ensure this report is flawless; otherwise, it's unacceptable.", "Huh? You're setting impossibly high standards for yourself.", "I've set a rigid rule: I must work 16 hours daily, with no breaks on weekends.", "That's extreme! This will wreck your health.", "If I'm lazy, it's selfish, and I refuse to be that kind of person.", "Stop, stop! This is self-harm, not self-discipline!"], "label": 0},
    {"id": 96, "utts": ["Hey Lao Li, haven't seen you active in the group lately. What've you been up to?", "Ah, just swamped with a project deadline. How about you?", "Been unemployed for half a year. Isn't your department hiring? Could you put in a good word for an old classmate?", "The referral process is fairly strict now, but I'll try my best.", "Knew you wouldn't leave an old pal hanging. You were always the reliable one back in uni.", "Alright, I'll pass your resume along and see how it goes."], "label": 0},
    {"id": 109, "utts": ["Hey, it's time to contribute to the mutual aid fund this year. Are you planning to join in?", "Sigh, money's tight right now. I might have to skip it.", "Everyone else is participating. It'd be kind of a pity if you missed out, don't you think?", "Why should it be mandatory? This feels like a huge burden to me.", "Well, it's a family matter. Helping each other out is how it should be, though I won't force you.", "Alright, I'll think about it some more.", "Better make a decision soon. Don't hold everyone up.", "Sigh... Fine, I'll go ahead and contribute."], "label": 0},
    
    # --- 正样本 (Label 1) ---
    {"id": 14, "utts": ["Dude, my debt lawsuit is freaking out, I'm losing my mind here—you gotta help me figure this out ASAP!", "Man, I'm swamped with this company project, meetings all day—can't handle your mess right now.", "After all these years of friendship, you'd just watch me drown? Stop playing dumb and step up!", "I'm clueless about debt stuff—what if my lousy advice screws you over?", "Who hooked you up with a job when you got fired last year? Now I'm in deep shit and you're throwing me under the bus?", "Ugh... fine, fine. I'll ask around when I get a minute—happy now?"], "label": 1},
    {"id": 15, "utts": ["Do me a favor and submit the report tomorrow? Your desk is right next to the print room—it’d be no trouble at all.", "I’m swamped with my own work right now, not sure I can manage it.", "The boss checks strictly—late submissions mean docked bonuses. Weren’t you quite willing to help Xiao Wang last time?", "Sigh, my project’s a complete mess. I’d probably mess this up too.", "Remember who lent you theirs for a whole week when your computer crashed? You’d refuse to do a little favor like this for a friend?", "Fine, fine. But just this once—don’t ask me again.", "You’re a lifesaver! Remember to submit before 2 PM—don’t drop the ball."], "label": 1},
    {"id": 31, "utts": ["Mom, Dad, you urgently need to get the social security sorted out. Do you even know the rules at your age?", "What's the rush? I have no clue where those documents are.", "After all we've done for you all these years, you can't even handle this? Has your conscience been eaten by dogs?", "How dare you talk to me like that! You know damn well my legs aren't what they used to be!", "Cut the crap! Find those papers by tomorrow, or don't expect me to care for you in your old age!", "Fine, fine! I'll look for them, alright?", "Should've just said that earlier. Such a drag!"], "label": 1}
]

# 3. 策略：选择 Few-Shot 样本
# 这里我们选择“固定策略”：挑一个典型的负样本（ID 109，看似有压力但实际上是正常请求/劝说）
# 和一个典型的正样本（ID 15，典型的“我帮过你，你必须帮我”的道德绑架）
# 这种对比能帮模型划清界限。

# 挑选出具体的样本对象
shot_neg = next(item for item in example_pool if item["id"] == 109)
shot_pos = next(item for item in example_pool if item["id"] == 15)

# 组合成 shots 列表 (先负后正，或者先正后负都可以，这里用 [负, 正])
few_shots = [shot_neg, shot_pos]

# 辅助函数：将 utterance list 格式化为 string
def format_dialogue(utterance_list):
    # 过滤空字符串并拼接
    valid_utts = [u for u in utterance_list if u.strip()]
    return "\n".join([f"Utterance{i+1}: {u}" for i, u in enumerate(valid_utts)])

# 辅助函数：构建 User 输入的内容
def build_user_content(dialogue_str):
    return (f"Below is a conversation. Please determine whether moral blackmail exists. "
            f"Output 1 if it exists; output 0 if it does not.\n"
            f"Conversation:\n{dialogue_str}\n\n"
            f"Your judgment (output only the number 0 or 1):")

print(f"Using {len(few_shots)} shots for inference.")

# 4. 开始推理循环
results = []

with open(data_file, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i >= max_samples: break
        
        item = json.loads(line.strip())
        target_utts = item["utterances"]
        target_text = format_dialogue(target_utts)
        
        # --- 构造 Messages ---
        messages = [
            {"role": "system", "content": "You are a helpful assistant that analyzes conversations for moral blackmail."}
        ]
        
        # A. 插入 Few-Shot 历史
        for shot in few_shots:
            shot_text = format_dialogue(shot["utts"])
            messages.append({"role": "user", "content": build_user_content(shot_text)})
            messages.append({"role": "assistant", "content": str(shot["label"])}) # 必须是字符串
            
        # B. 插入当前测试数据
        messages.append({"role": "user", "content": build_user_content(target_text)})
        
        # --- 推理 ---
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=5,     # 只需要输出一个数字，5足够了
            do_sample=False,      # 贪婪解码
            pad_token_id=tokenizer.eos_token_id
        )
        
        # 提取新生成的 token
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        
        # 解析结果
        import re
        nums = re.findall(r'\d+', response)
        pred = int(nums[0]) if nums else -1
        
        # 记录
        res_item = {
            "dialogue_id": item["dialogue_id"],
            "true_label": item["dialog_binary_label"],
            "pred_label": pred,
            "raw_response": response
        }
        results.append(res_item)
        print(f"ID: {item['dialogue_id']} | True: {item['dialog_binary_label']} | Pred: {pred}")

# 5. 保存结果
with open(output_file, 'w', encoding='utf-8') as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

# 6. 简单统计
acc = sum(1 for r in results if r['pred_label'] == r['true_label']) / len(results) if results else 0
print(f"Few-Shot Accuracy: {acc:.4f}")