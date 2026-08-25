import json
import numpy as np
from sklearn.metrics import cohen_kappa_score

def evaluate_ordinal_metrics(file_path):
    y_true = []
    y_pred = []
    
    # 1. 读取 jsonl 文件
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            y_true.append(int(data['ground_truth']))
            y_pred.append(int(data['prediction']))
            
    # 转换为 numpy 数组以支持矢量化运算
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    total_samples = len(y_true)
    
    # 2. 计算预测值与真实值之间的绝对距离 (Absolute Distance)
    # 例如：真实是 4，预测是 3，距离就是 1
    diff = np.abs(y_true - y_pred)
    
    # 3. 计算各种准确率
    # 严格准确率 (Distance == 0)
    strict_acc = np.mean(diff == 0)
    
    # 相邻准确率 / Accuracy ± 1 (Distance <= 1)
    adj_acc_1 = np.mean(diff <= 1)
    
    # 宽容准确率 / Accuracy ± 2 (Distance <= 2) - 用于辅助分析
    adj_acc_2 = np.mean(diff <= 2)
    
    # 4. 计算二次加权 Kappa (Quadratic Weighted Kappa)
    # weights='quadratic' 会对相差较远的错误施加平方级的惩罚
    qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')
    
    # ================= 打印报告 =================
    print(f"=== 有序分类/强度评级评估报告 (Ordinal Evaluation) ===")
    print(f"总测试样本数: {total_samples}")
    print("-" * 50)
    print(f"严格准确率 (Strict Accuracy): {strict_acc:.4f}  ({strict_acc*100:.2f}%)")
    print(f"相邻准确率 (Accuracy ± 1):     {adj_acc_1:.4f}  ({adj_acc_1*100:.2f}%)")
    print(f"二次加权 Kappa (QWK):          {qwk:.4f}")
    print("-" * 50)
    
    # 5. 打印误差分布 (对写论文的 Error Analysis 非常有帮助)
    print("\n=== 预测误差距离分布 (Error Distance Distribution) ===")
    print("距离 | 样本数 | 占比")
    for i in range(6):  # 最大距离为 5 (预测0真实5，或相反)
        count = np.sum(diff == i)
        ratio = count / total_samples
        if count > 0:
            print(f"  {i}  |  {count:4d}  | {ratio*100:.2f}%")

if __name__ == "__main__":
    # 请替换为你的 Fine-tuned Qwen 或 ChatGLM 的结果文件路径
    # file_path = 'save/c_model/fine-tuning/ChatGLM/Coercion-47K/multi_results.jsonl'
    file_path = 'save/c_model/fine-tuning/Llama/7B/multi/Llama7_LoRA/results.jsonl'

    
    evaluate_ordinal_metrics(file_path)