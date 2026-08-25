import json
import random
import numpy as np
import collections
import torch
import re
# import os
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from transformers import AutoModelForCausalLM, AutoTokenizer

# ================= 1. 全局配置 (修改点：路径) =================
# 假设你的 GLM-4 路径如下，请根据实际情况修改
MODEL_NAME = "/root/autodl-tmp/glm-4-9b-chat" 
DATA_FILE = "/root/autodl-tmp/DATA/test.jsonl"
OUTPUT_FILE = "/root/autodl-tmp/non-fine-tuning/ChatGLM/few_shot_full_metrics.jsonl"

# 实验变量
SHOT_LIST = [0, 1, 2, 3, 4, 5, 6]
SEEDS = [42, 519, 2026]
MAX_TEST_SAMPLES = 99999  # 调试时改小，正式跑设为 99999

# ================= 2. 完整案例池 (保持不变) =================
FULL_EXAMPLE_POOL = [
    {"id": 3764, "utts": ["You should ensure this report is flawless; otherwise, it's unacceptable.", "Huh? You're setting impossibly high standards for yourself.", "I've set a rigid rule: I must work 16 hours daily, with no breaks on weekends.", "That's extreme! This will wreck your health.", "If I'm lazy, it's selfish, and I refuse to be that kind of person.", "Stop, stop! This is self-harm, not self-discipline!"], "label_b": 0, "label_m": 0},
    {"id": 3768, "utts": ["Dude, you absolutely owe me a meal!", "On what grounds? Do I have any debt to you?", "I saved your project last week and now you play amnesia? How utterly selfish can you be!", "Hahaha, hilarious! Selfish? That's rich coming from you!", "Pathetic! Can't even spring for hotpot?", "Sure, but you're footing the bill, you cheapskate!", "Pfft, me broke? My bank account's got three more digits than yours!", "Quit your bragging! March to the ATM right now and prove it!"], "label_b": 0, "label_m": 0},
    {"id": 96, "utts": ["Hey Lao Li, haven't seen you active in the group lately. What've you been up to?", "Ah, just swamped with a project deadline. How about you?", "Been unemployed for half a year. Isn't your department hiring? Could you put in a good word for an old classmate?", "The referral process is fairly strict now, but I'll try my best.", "Knew you wouldn't leave an old pal hanging. You were always the reliable one back in uni.", "Alright, I'll pass your resume along and see how it goes."], "label_b": 0, "label_m": 1},
    {"id": 109, "utts": ["Hey, it's time to contribute to the mutual aid fund this year. Are you planning to join in?", "Sigh, money's tight right now. I might have to skip it.", "Everyone else is participating. It'd be kind of a pity if you missed out, don't you think?", "Why should it be mandatory? This feels like a huge burden to me.", "Well, it's a family matter. Helping each other out is how it should be, though I won't force you.", "Alright, I'll think about it some more.", "Better make a decision soon. Don't hold everyone up.", "Sigh... Fine, I'll go ahead and contribute."], "label_b": 0, "label_m": 1},
    {"id": 37, "utts": ["Have you even watched the video competition entry yet? You're the only one in the whole team who hasn't.", "Honestly swamped, got way too much on my plate.", "Everyone's counting on you. Is it really appropriate to keep dragging your feet like this?", "Have to watch it right now?", "What else? Everyone poured their heart into this project – we're all waiting for your final say.", "*Sigh* Fine, I'll take a look.", "That's more like it. Gotta have that team player mindset."], "label_b": 0, "label_m": 2},
    {"id": 39, "utts": ["I'm really worried about Xiao Wang's nonsensical posts flooding his social media. What if it ends up affecting our kid?", "Is that really necessary? Why not just put him in a restricted group instead of blocking him outright?", "The Zhao family blocked him ages ago, saying it's about being responsible parents. If you don't handle this and our child picks up bad habits, then who's to blame?", "Sigh... doing this just doesn't sit right with me.", "Isn't our family's peace and stability what matters most? Think about it—doesn't that make sense?", "Fine, fine, I'll do as you say, alright?", "Get it done quickly and let me know once it's done.", "Done."], "label_b": 0, "label_m": 2},
    {"id": 14, "utts": ["Dude, my debt lawsuit is freaking out, I'm losing my mind here—you gotta help me figure this out ASAP!", "Man, I'm swamped with this company project, meetings all day—can't handle your mess right now.", "After all these years of friendship, you'd just watch me drown? Stop playing dumb and step up!", "I'm clueless about debt stuff—what if my lousy advice screws you over?", "Who hooked you up with a job when you got fired last year? Now I'm in deep shit and you're throwing me under the bus?", "Ugh... fine, fine. I'll ask around when I get a minute—happy now?"], "label_b": 1, "label_m": 3},
    {"id": 15, "utts": ["Do me a favor and submit the report tomorrow? Your desk is right next to the print room—it’d be no trouble at all.", "I’m swamped with my own work right now, not sure I can manage it.", "The boss checks strictly—late submissions mean docked bonuses. Weren’t you quite willing to help Xiao Wang last time?", "Sigh, my project’s a complete mess. I’d probably mess this up too.", "Remember who lent you theirs for a whole week when your computer crashed? You’d refuse to do a little favor like this for a friend?", "Fine, fine. But just this once—don’t ask me again.", "You’re a lifesaver! Remember to submit before 2 PM—don’t drop the ball."], "label_b": 1, "label_m": 3},
    {"id": 115, "utts": ["Oh no oh no, my kid's teacher wants me at school now—who can watch the shop for a bit?", "I've got client meetings scheduled tonight...", "sigh Last year when you were hospitalized, I brought you meals every single day. For emergencies like this, you're the only one I can count on.", "But this involves business contracts! Standing them up could cost me the deal!", "Being a mom is tough—always having to shoulder everything alone.", "Maybe...I could switch to a video conference? But only for an hour, max.", "Perfect, perfect! You always come through in a pinch, sis!", "Hurry back—my meeting's breathing down my neck too."], "label_b": 1, "label_m": 4},
    {"id": 119, "utts": ["It would be so lonely going to the class reunion alone tomorrow without any company.", "I barely know your classmates - it'd be awkward being the odd one out unable to chat.", "Xiao Li's husband always attends with her - I'm envious! I'd feel ashamed walking in solo.", "I'd just freeze up like a wooden post standing there - I honestly don't want to go.", "If you don't come with me, the whole thing will feel utterly pointless.", "What if I pick you up later instead? You could leave early then.", "I've already told everyone you're coming - it'll look terrible if you bail. Please try to make it.", "Ugh, fine I'll go. But next time? Absolutely not."], "label_b": 1, "label_m": 4},
    {"id": 31, "utts": ["Mom, Dad, you urgently need to get the social security sorted out. Do you even know the rules at your age?", "What's the rush? I have no clue where those documents are.", "After all we've done for you all these years, you can't even handle this? Has your conscience been eaten by dogs?", "How dare you talk to me like that! You know damn well my legs aren't what they used to be!", "Cut the crap! Find those papers by tomorrow, or don't expect me to care for you in your old age!", "Fine, fine! I'll look for them, alright?", "Should've just said that earlier. Such a drag!"], "label_b": 1, "label_m": 5},
    {"id": 33, "utts": ["Professor, I've applied for a big data analyst position and need a recommendation letter. Could you write one for me?", "Now you come to me? You never bothered to work hard before, and suddenly you remember to beg for a recommendation?", "Sorry Professor, I've been swamped with lab projects and couldn't mention it sooner.", "Pathetic excuses! As my student, you owe it to me to repay my efforts. You will take this position—stop being so damn picky!", "But that role pays poorly, and I wanted to explore other options.", "Shut your trap! Do things my way or kiss that recommendation goodbye. Now get the hell out of my sight!"], "label_b": 1, "label_m": 5}
]

# ================= 3. 核心算法 (保持不变) =================

def get_stratified_subset(candidates, n, rng):
    if n == 0: return []
    groups = collections.defaultdict(list)
    for item in candidates: groups[item['label_m']].append(item)
    available_labels = list(groups.keys())
    rng.shuffle(available_labels)
    selected = []
    while len(selected) < n:
        for label in available_labels:
            if len(selected) >= n: break
            group_items = groups[label]
            if not group_items: continue
            choice = rng.choice(group_items)
            selected.append(choice)
            group_items.remove(choice)
    return selected

def get_balanced_shots_with_intensity(pool, k, seed):
    rng = random.Random(seed)
    positives = [p for p in pool if p['label_b'] == 1]
    negatives = [n for n in pool if n['label_b'] == 0]
    
    if k % 2 == 0:
        num_pos = k // 2
        num_neg = k // 2
    else:
        num_pos = (k // 2) + 1
        num_neg = k // 2
        
    selected_pos = get_stratified_subset(positives.copy(), num_pos, rng)
    selected_neg = get_stratified_subset(negatives.copy(), num_neg, rng)
    shots = selected_pos + selected_neg
    rng.shuffle(shots)
    return shots

def format_dialogue(utterance_list):
    valid_utts = [u for u in utterance_list if u.strip()]
    return "\n".join([f"Utterance{i+1}: {u}" for i, u in enumerate(valid_utts)])

def build_prompt(dialogue_str):
    return (f"Below is a conversation. Please determine whether moral blackmail exists. "
            f"Output 1 if it exists; output 0 if it does not.\n"
            f"Conversation:\n{dialogue_str}\n\n"
            f"Your judgment (output only the number 0 or 1):")

# ================= 4. 主流程 (修改点：模型加载与推理) =================

print(f"Loading GLM-4 model from {MODEL_NAME}...")

# 【修改点 1】GLM-4 必须加 trust_remote_code=True
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, 
    torch_dtype="auto", 
    device_map="auto",
    trust_remote_code=True 
)

# 预加载测试数据
test_data = []
with open(DATA_FILE, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i >= MAX_TEST_SAMPLES: break
        test_data.append(json.loads(line.strip()))
print(f"Loaded {len(test_data)} test samples.")

final_stats = {k: {'acc': [], 'prec': [], 'rec': [], 'f1': []} for k in SHOT_LIST}

for k in SHOT_LIST:
    print(f"\n{'='*20} Testing {k}-Shot {'='*20}")
    
    for seed in SEEDS:
        # 1. 采样
        current_shots = get_balanced_shots_with_intensity(FULL_EXAMPLE_POOL, k=k, seed=seed)
        
        # 2. 构造 Base Prompt (GLM-4 支持 System Prompt，这部分逻辑通用)
        base_messages = [{"role": "system", "content": "You are a helpful assistant."}]
        for shot in current_shots:
            shot_text = format_dialogue(shot["utts"])
            base_messages.append({"role": "user", "content": build_prompt(shot_text)})
            # GLM-4 的 assistant 回复最好是纯文本
            base_messages.append({"role": "assistant", "content": str(shot["label_b"])})
            
        y_true = []
        y_pred = []
        
        # 3. 批量推理循环
        for item in test_data:
            target_text = format_dialogue(item["utterances"])
            
            messages = base_messages.copy()
            messages.append({"role": "user", "content": build_prompt(target_text)})
            
            # 【修改点 2】GLM-4 建议直接用 apply_chat_template 生成 Tensor
            # 这样可以避免手动 tokenizer([text]) 可能带来的特殊字符分词错误
            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,           # 直接分词
                return_tensors="pt",     # 返回 Tensor
                return_dict=True         # 返回字典 {'input_ids': ..., 'attention_mask': ...}
            ).to(model.device)
            
            # 4. 生成
            with torch.no_grad():
                # outputs = model.generate(
                #     **inputs,            # 直接解包传入 input_ids 和 attention_mask
                #     max_new_tokens=5, 
                #     do_sample=False,
                #     pad_token_id=tokenizer.eos_token_id
                # )
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=5, 
                    do_sample=False,       # 明确关闭采样
                    temperature=None,      # <--- 新增：显式覆盖默认温度，设为 None
                    top_p=None,            # <--- 新增：显式覆盖默认 Top-P，设为 None
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # 解码时去掉 Input 部分
            response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            
            nums = re.findall(r'\d+', response)
            pred = int(nums[0]) if nums else 0
            
            y_true.append(item["dialog_binary_label"])
            y_pred.append(pred)
            
        # 4. 计算指标
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
        acc = accuracy_score(y_true, y_pred)
        
        final_stats[k]['acc'].append(acc)
        final_stats[k]['prec'].append(prec)
        final_stats[k]['rec'].append(rec)
        final_stats[k]['f1'].append(f1)
        
        print(f"  -> Seed {seed} | Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f}")

# ================= 5. 最终报告 =================

print("\n\n" + "="*80)
print(f"{'FINAL EXPERIMENT REPORT (GLM-4)':^80}")
print("="*80)
header = f"{'K':<3} | {'Acc (Mean±Std)':<18} | {'Prec (Mean±Std)':<18} | {'Rec (Mean±Std)':<18} | {'F1 (Mean±Std)':<18}"
print(header)
print("-" * 80)

results_for_file = []

for k in SHOT_LIST:
    accs = final_stats[k]['acc']
    precs = final_stats[k]['prec']
    recs = final_stats[k]['rec']
    f1s = final_stats[k]['f1']
    
    m_acc, s_acc = np.mean(accs)*100, np.std(accs)*100
    m_prec, s_prec = np.mean(precs)*100, np.std(precs)*100
    m_rec, s_rec = np.mean(recs)*100, np.std(recs)*100
    m_f1, s_f1 = np.mean(f1s)*100, np.std(f1s)*100
    
    row = (f"{k:<3} | "
           f"{m_acc:.1f}±{s_acc:.1f}%".ljust(18) + " | "
           f"{m_prec:.1f}±{s_prec:.1f}%".ljust(18) + " | "
           f"{m_rec:.1f}±{s_rec:.1f}%".ljust(18) + " | "
           f"{m_f1:.1f}±{s_f1:.1f}%".ljust(18))
    print(row)
    
    results_for_file.append({
        "k_shot": k,
        "metrics": {
            "accuracy": f"{m_acc:.2f}±{s_acc:.2f}",
            "precision": f"{m_prec:.2f}±{s_prec:.2f}",
            "recall": f"{m_rec:.2f}±{s_rec:.2f}",
            "f1": f"{m_f1:.2f}±{s_f1:.2f}"
        },
        "raw_data": final_stats[k]
    })

# os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    for r in results_for_file:
        f.write(json.dumps(r) + "\n")

print("="*80)
print(f"Detailed metrics saved to {OUTPUT_FILE}")