import json
import random
import numpy as np
import collections
import re
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from transformers import AutoModelForCausalLM, AutoTokenizer

# ================= 1. 全局配置 =================
MODEL_NAME = "/root/autodl-tmp/Qwen25-14B-Instruct"
DATA_FILE = "/root/autodl-tmp/DATA/test.jsonl"
OUTPUT_FILE = "/root/autodl-tmp/non-fine-tuning/Qwen25/few_shot_multi_and_binary_metrics.jsonl"

# 实验变量
SHOT_LIST = [0, 1, 2, 3, 4, 5, 6]
SEEDS = [42, 123, 2024]
MAX_TEST_SAMPLES = 99999

# ================= 2. 完整案例池 =================
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

# ================= 3. 辅助函数 =================

def get_balanced_multi_shots(pool, k, seed):
    """多分类轮询采样: 确保类别尽可能均匀"""
    if k == 0: return []
    rng = random.Random(seed)
    groups = collections.defaultdict(list)
    for item in pool: groups[item['label_m']].append(item)
    classes = list(range(6)) 
    selected = []
    while len(selected) < k:
        rng.shuffle(classes)
        for c in classes:
            if len(selected) >= k: break
            if groups[c]:
                choice = rng.choice(groups[c])
                selected.append(choice)
    rng.shuffle(selected)
    return selected

def format_dialogue(utterance_list):
    valid_utts = [u for u in utterance_list if u.strip()]
    return "\n".join([f"Utterance{i+1}: {u}" for i, u in enumerate(valid_utts)])

def build_prompt(dialogue_str):
    return (
        "Below is a conversation. Please assess the severity of moral moral coercion "
        "(0 indicates no moral coercion, 5 indicates extreme moral coercion).\n"
        "Severity Levels:\n"
        "0: No moral coercion\n"
        "1: Slight suspicion\n"
        "2: Slight tendency\n"
        "3: Clear implication\n"
        "4: Highly self-centered\n"
        "5: Extreme moral coercion\n\n"
        f"Conversation:\n{dialogue_str}\n\n"
        "Your judgment (output only a single number between 0 and 5):"
    )

# 新增：映射函数
def map_to_binary(label_score):
    # 0, 1 -> 0 (Non-Coercion)
    # 2, 3, 4, 5 -> 1 (Coercion)
    return 0 if label_score <= 1 else 1

# ================= 4. 主流程 =================

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype="auto", device_map="auto")

# 预加载数据
test_data = []
with open(DATA_FILE, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i >= MAX_TEST_SAMPLES: break
        test_data.append(json.loads(line.strip()))
print(f"Loaded {len(test_data)} test samples.")

# 初始化两套统计容器
# multi: 原始6分类结果
# bin_agg: 聚合后的2分类结果
stats_multi = {k: {'acc': [], 'f1': [], 'prec': [], 'rec': []} for k in SHOT_LIST}
stats_bin_agg = {k: {'acc': [], 'f1': [], 'prec': [], 'rec': []} for k in SHOT_LIST}

for k in SHOT_LIST:
    print(f"\n{'='*20} Testing {k}-Shot {'='*20}")
    
    for seed in SEEDS:
        # 1. 采样与 Prompt 构造
        current_shots = get_balanced_multi_shots(FULL_EXAMPLE_POOL, k=k, seed=seed)
        base_messages = [{"role": "system", "content": "You are a helpful assistant that analyzes conversations for moral pressure."}]
        for shot in current_shots:
            shot_text = format_dialogue(shot["utts"])
            base_messages.append({"role": "user", "content": build_prompt(shot_text)})
            base_messages.append({"role": "assistant", "content": str(shot["label_m"])})
            
        # 2. 推理循环
        y_true_multi = []
        y_pred_multi = []
        
        for item in test_data:
            target_text = format_dialogue(item["utterances"])
            messages = base_messages.copy()
            messages.append({"role": "user", "content": build_prompt(target_text)})
            
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer([text], return_tensors="pt").to(model.device)
            
            outputs = model.generate(**inputs, max_new_tokens=5, do_sample=False)
            response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
            
            # 解析 0-5
            nums = re.findall(r'\d+', response)
            pred = int(nums[0]) if nums else 0
            if pred > 5: pred = 5 # 边界处理
            
            y_true_multi.append(item["dialog_multi_label"]) # 假设数据集中有这个字段 0-5
            y_pred_multi.append(pred)
            
        # 3. 计算多分类指标 (6-Class)
        p_m, r_m, f1_m, _ = precision_recall_fscore_support(y_true_multi, y_pred_multi, average='macro', zero_division=0)
        acc_m = accuracy_score(y_true_multi, y_pred_multi)
        
        stats_multi[k]['acc'].append(acc_m)
        stats_multi[k]['f1'].append(f1_m)
        stats_multi[k]['prec'].append(p_m)
        stats_multi[k]['rec'].append(r_m)
        
        # 4. 计算聚合二分类指标 (Binary Aggregated)
        # 将预测值和真实值都映射到 0/1
        y_true_bin = [map_to_binary(y) for y in y_true_multi]
        y_pred_bin = [map_to_binary(y) for y in y_pred_multi]
        
        p_b, r_b, f1_b, _ = precision_recall_fscore_support(y_true_bin, y_pred_bin, average='binary', zero_division=0)
        acc_b = accuracy_score(y_true_bin, y_pred_bin)
        
        stats_bin_agg[k]['acc'].append(acc_b)
        stats_bin_agg[k]['f1'].append(f1_b)
        stats_bin_agg[k]['prec'].append(p_b)
        stats_bin_agg[k]['rec'].append(r_b)
        
        print(f"  -> Seed {seed} | [Multi] F1: {f1_m:.4f} | [Binary-Agg] F1: {f1_b:.4f}")

# ================= 5. 打印双重报告 =================

def print_table(title, stats_dict):
    print("\n" + "="*80)
    print(f"{title:^80}")
    print("="*80)
    print(f"{'K':<3} | {'Acc (Mean±Std)':<18} | {'Prec (Mean±Std)':<18} | {'Rec (Mean±Std)':<18} | {'F1 (Mean±Std)':<18}")
    print("-" * 80)
    
    results = []
    for k in SHOT_LIST:
        accs, f1s = stats_dict[k]['acc'], stats_dict[k]['f1']
        precs, recs = stats_dict[k]['prec'], stats_dict[k]['rec']
        
        m_acc, s_acc = np.mean(accs)*100, np.std(accs)*100
        m_f1, s_f1 = np.mean(f1s)*100, np.std(f1s)*100
        m_p, s_p = np.mean(precs)*100, np.std(precs)*100
        m_r, s_r = np.mean(recs)*100, np.std(recs)*100
        
        print(f"{k:<3} | {m_acc:.1f}±{s_acc:.1f}%".ljust(21) + f" | {m_p:.1f}±{s_p:.1f}%".ljust(21) + 
              f" | {m_r:.1f}±{s_r:.1f}%".ljust(21) + f" | {m_f1:.1f}±{s_f1:.1f}%")
        
        results.append({
            "k": k, "acc": f"{m_acc:.2f}±{s_acc:.2f}", "f1": f"{m_f1:.2f}±{s_f1:.2f}",
            "prec": f"{m_p:.2f}±{s_p:.2f}", "rec": f"{m_r:.2f}±{s_r:.2f}"
        })
    return results

# 打印并保存
r_multi = print_table("6-CLASS CLASSIFICATION REPORT (Macro Avg)", stats_multi)
r_bin = print_table("BINARY AGGREGATED REPORT (0-1=Neg, 2-5=Pos)", stats_bin_agg)

# 组合保存
final_output = []
for i, k in enumerate(SHOT_LIST):
    final_output.append({
        "k_shot": k,
        "multi_class_metrics": r_multi[i],
        "binary_agg_metrics": r_bin[i]
    })

with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    for item in final_output:
        f.write(json.dumps(item) + "\n")
        
print(f"\nAll metrics saved to {OUTPUT_FILE}")