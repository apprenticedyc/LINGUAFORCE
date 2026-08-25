import os
import json
import random
import numpy as np
import collections
import re
import torch
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from transformers import AutoModelForCausalLM, AutoTokenizer

# ================= 1. 全局配置 =================
MODEL_NAME = "/root/autodl-tmp/Llama-2-7b-chat-ms" 
DATA_FILE = "/root/autodl-tmp/DATA/test.jsonl"
OUTPUT_BASE_DIR = "/root/autodl-tmp/non-fine-tuning/Llama2"
SUMMARY_FILE = os.path.join(OUTPUT_BASE_DIR, "few_shot_full_metrics.jsonl")

# 实验变量
SHOT_LIST = [0, 1, 2, 3, 4, 5, 6]  # 测试的 Shot 数量
SEEDS = [42, 519, 2026]         # 3 个种子
MAX_TEST_SAMPLES = 99999        # 调试时改小     

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

# ================= 2. 初始化环境 =================
os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
for seed in SEEDS:
    os.makedirs(os.path.join(OUTPUT_BASE_DIR, f"seed_{seed}"), exist_ok=True)

# ================= 3. 核心算法 =================
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
    num_pos = (k // 2) + 1 if k % 2 != 0 else k // 2
    num_neg = k // 2
    selected_pos = get_stratified_subset(positives.copy(), num_pos, rng)
    selected_neg = get_stratified_subset(negatives.copy(), num_neg, rng)
    shots = selected_pos + selected_neg
    rng.shuffle(shots)
    return shots

def format_dialogue(utterance_list):
    valid_utts = [u for u in utterance_list if u.strip()]
    return "\n".join([f"Utterance{i+1}: {u}" for i, u in enumerate(valid_utts)])

# ================= 4. 主流程 (Raw Prompting 无模板模式) =================

print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 使用半精度加载以提升性能
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, 
    torch_dtype=torch.float16, 
    device_map="auto"
)

test_data = []
with open(DATA_FILE, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i >= MAX_TEST_SAMPLES: break
        test_data.append(json.loads(line.strip()))

final_stats = {k: {'acc': [], 'prec': [], 'rec': [], 'f1': []} for k in SHOT_LIST}

for k in SHOT_LIST:
    print(f"\n{'='*20} Testing {k}-Shot {'='*20}")
    
    for seed in SEEDS:
        current_shots = get_balanced_shots_with_intensity(FULL_EXAMPLE_POOL, k=k, seed=seed)
        
        y_true = []
        y_pred = []
        current_run_predictions = [] 
        
        for item in test_data:
            target_text = format_dialogue(item["utterances"])
            
            # --- 构建纯文本 Prompt ---
            # 直接通过换行和 Output 引导模型进行填空
            raw_prompt = "Below is a conversation. Please determine whether moral blackmail exists. Output strictly 1 for Yes, or 0 for No.\n\n"
            
            # 拼接 Few-shot 示例
            if k > 0:
                for i, shot in enumerate(current_shots):
                    shot_text = format_dialogue(shot["utts"])
                    raw_prompt += f"Conversation:\n{shot_text}\nOutput: {shot['label_b']}\n\n"
            
            # 拼接测试目标
            raw_prompt += f"Conversation:\n{target_text}\nOutput:"
            
            # --- 编码与推理 ---
            inputs = tokenizer(raw_prompt, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=5, # 填空模式下，5 个 token 绰绰有余
                    do_sample=False   # 保持结果稳定
                )
                
            # 只解码新生成的部分
            response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
            
            # 正则提取第一个独立数字
            match = re.search(r'\b(0|1)\b', response)
            pred = int(match.group(1)) if match else 0
            
            y_true.append(item["dialog_binary_label"])
            y_pred.append(pred)
            
            dialogue_id = item.get("id", item.get("dialogue_id", "unknown"))
            current_run_predictions.append({
                "dialogue_id": dialogue_id,
                "true_binary": item["dialog_binary_label"],
                "pred_binary": pred,
                "model_response": response.strip()
            })
            
        # 计算指标
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
        acc = accuracy_score(y_true, y_pred)
        
        final_stats[k]['acc'].append(acc)
        final_stats[k]['prec'].append(prec)
        final_stats[k]['rec'].append(rec)
        final_stats[k]['f1'].append(f1)
        
        # 保存结果
        shot_file_path = os.path.join(OUTPUT_BASE_DIR, f"seed_{seed}", f"{k}_shot.jsonl")
        with open(shot_file_path, 'w', encoding='utf-8') as pf:
            for record in current_run_predictions:
                pf.write(json.dumps(record) + "\n")
                
        print(f"  -> Seed {seed} | Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f}")

# ================= 5. 输出汇总报告 =================
print("\n\n" + "="*80)
print(f"{'FINAL EXPERIMENT REPORT':^80}")
print("="*80)
header = f"{'K':<3} | {'Acc (Mean±Std)':<18} | {'Prec (Mean±Std)':<18} | {'Rec (Mean±Std)':<18} | {'F1 (Mean±Std)':<18}"
print(header)
print("-" * 80)

results_for_file = []
for k in SHOT_LIST:
    accs, precs, recs, f1s = final_stats[k]['acc'], final_stats[k]['prec'], final_stats[k]['rec'], final_stats[k]['f1']
    m_acc, s_acc = np.mean(accs)*100, np.std(accs)*100
    m_prec, s_prec = np.mean(precs)*100, np.std(precs)*100
    m_rec, s_rec = np.mean(recs)*100, np.std(recs)*100
    m_f1, s_f1 = np.mean(f1s)*100, np.std(f1s)*100
    
    row = (f"{k:<3} | {m_acc:.1f}±{s_acc:.1f}%".ljust(24) + f" | {m_prec:.1f}±{s_prec:.1f}%".ljust(21) + 
           f" | {m_rec:.1f}±{s_rec:.1f}%".ljust(21) + f" | {m_f1:.1f}±{s_f1:.1f}%")
    print(row)
    
    results_for_file.append({
        "k_shot": k,
        "metrics": {"accuracy": f"{m_acc:.2f}±{s_acc:.2f}", "precision": f"{m_prec:.2f}±{s_prec:.2f}",
                    "recall": f"{m_rec:.2f}±{s_rec:.2f}", "f1": f"{m_f1:.2f}±{s_f1:.2f}"},
        "raw_data": final_stats[k]
    })

with open(SUMMARY_FILE, 'w', encoding='utf-8') as f:
    for r in results_for_file:
        f.write(json.dumps(r) + "\n")
print(f"\nSummary metrics saved to: {SUMMARY_FILE}")