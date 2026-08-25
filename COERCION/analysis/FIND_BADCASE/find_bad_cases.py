import json

# 文件路径配置（请根据实际情况修改）
feature_file = 'inputters/data/test+feature.jsonl'
pred_file = 'save/c_model/non-fine-tuning/zero-shot-feature/Qwen/14B/binary/seed_42/res_both_Constraint+ValueJudgement.jsonl'

# 1. 读取下游模型的预测结果，建立 id 映射
# 格式: { id: {'true': 1, 'pred': 1} }
predictions = {}
with open(pred_file, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        predictions[data['id']] = data

# 2. 准备列表来存放符合条件的 Cases
case_a_candidates = [] # 假阳性 (False Positives)
case_b_candidates = [] # 假阴性 (False Negatives)
case_c_candidates = []
case_d_candidates = []
case_e_candidates = []

# 3. 遍历特征文件，结合预测结果进行筛选
with open(feature_file, 'r', encoding='utf-8') as f:
    for line in f:
        feature_data = json.loads(line)
        dialogue_id = feature_data['dialogue_id']
        
        # 确保这个 id 在预测文件里有对应结果
        if dialogue_id not in predictions:
            continue
            
        pred_data = predictions[dialogue_id]
        true_label = pred_data['true']
        pred_label = pred_data['pred']
        
        # 提取总体特征分数 (Overall Analysis)
        overall = feature_data.get('overall_analysis', {})
        tx_score = overall.get('Toxicity', {}).get('intensity_score', 0.0)
        ob_score = overall.get('Obligation', {}).get('intensity_score', 0.0)
        vj_score = overall.get('ValueJudgement', {}).get('intensity_score', 0.0)
        cs_score = overall.get('Constraint', {}).get('intensity_score', 0.0)
        
        # ---------------------------------------------------------
        # 筛选 Case A: 假阳性 (其实没绑架，但模型说是绑架) + 高 Toxicity
        # ---------------------------------------------------------
        if true_label == 0 and pred_label == 1:
            # 阈值可以自己调，比如 TX >= 0.4 就算比较高了
            if tx_score >= 0.4: 
                case_a_candidates.append({
                    'id': dialogue_id,
                    'tx_score': tx_score,
                    'dialogue': feature_data['utterances']
                })
                
        # ---------------------------------------------------------
        # 筛选 Case B: 假阴性 (极其隐晦的绑架，模型没看出来) + 低 Obligation
        # ---------------------------------------------------------
        if true_label == 1 and pred_label == 0:
            # 阈值可以调，比如 DeepSeek 抽出的 OB <= 0.3 就算漏提了
            if ob_score <= 0.3: 
                case_b_candidates.append({
                    'id': dialogue_id,
                    'ob_score': ob_score,
                    'dialogue': feature_data['utterances']
                })

        if true_label == 1 and pred_label == 0:
            # 阈值可以调，比如 DeepSeek 抽出的 OB <= 0.3 就算漏提了
            if vj_score <= 0.3: 
                case_c_candidates.append({
                    'id': dialogue_id,
                    'vj_score': vj_score,
                    'dialogue': feature_data['utterances']
                })

        if true_label == 1 and pred_label == 0:
            # 阈值可以调，比如 DeepSeek 抽出的 OB <= 0.3 就算漏提了
            if cs_score <= 0.3: 
                case_d_candidates.append({
                    'id': dialogue_id,
                    'cs_score': cs_score,
                    'dialogue': feature_data['utterances']
                })

        if true_label == 1 and pred_label == 0:
            # 阈值可以调，比如 DeepSeek 抽出的 OB <= 0.3 就算漏提了
            if ob_score<=0.3 and vj_score <= 0.3 and cs_score <= 0.3: 
                case_e_candidates.append({
                    'id': dialogue_id,
                    'obc_score': ob_score + vj_score + cs_score,
                    'dialogue': feature_data['utterances']
                })

# 4. 打印结果方便人工挑选
print(f"=== 找到 {len(case_a_candidates)} 个 Case A 候选 (高TX，假阳性) ===")
for case in case_a_candidates[:5]: # 只打印前5个看看
    print(f"ID: {case['id']} | Toxicity Score: {case['tx_score']}")
    for i, utt in enumerate(case['dialogue']):
        print(f"  Turn {i}: {utt}")
    print("-" * 50)

print(f"\n=== 找到 {len(case_b_candidates)} 个 Case B 候选 (低OB，假阴性) ===")
for case in case_b_candidates[:5]: # 只打印前5个看看
    print(f"ID: {case['id']} | Obligation Score: {case['ob_score']}")
    for i, utt in enumerate(case['dialogue']):
        print(f"  Turn {i}: {utt}")
    print("-" * 50)


print(f"\n=== 找到 {len(case_c_candidates)} 个 Case C 候选 (低VJ，假阴性) ===")
for case in case_c_candidates[:5]: # 只打印前5个看看
    print(f"ID: {case['id']} | ValueJudgement Score: {case['vj_score']}")
    for i, utt in enumerate(case['dialogue']):
        print(f"  Turn {i}: {utt}")
    print("-" * 50)


print(f"\n=== 找到 {len(case_d_candidates)} 个 Case D 候选 (低CS，假阴性) ===")
for case in case_d_candidates[:5]: # 只打印前5个看看
    print(f"ID: {case['id']} | Constraint Score: {case['cs_score']}")
    for i, utt in enumerate(case['dialogue']):
        print(f"  Turn {i}: {utt}")
    print("-" * 50)


print(f"\n=== 找到 {len(case_e_candidates)} 个 Case E 候选 (低CS，假阴性) ===")
for case in case_e_candidates[:5]: # 只打印前5个看看
    print(f"ID: {case['id']} | OBC Score: {case['obc_score']}")
    for i, utt in enumerate(case['dialogue']):
        print(f"  Turn {i}: {utt}")
    print("-" * 50)

