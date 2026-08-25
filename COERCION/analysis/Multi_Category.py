import json
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

def evaluate_predictions(file_path):
    y_true = []
    y_pred = []
    
    # 1. 读取 jsonl 文件
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            y_true.append(data['ground_truth'])
            y_pred.append(data['prediction'])
            
    # 转换为 numpy 数组方便后续计算
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # 获取所有的类别标签 (假设是 0, 1, 2, 3, 4, 5)
    labels = sorted(list(set(y_true) | set(y_pred)))
    
    # 2. 计算总体准确率 (Overall Accuracy)
    overall_acc = accuracy_score(y_true, y_pred)
    print(f"=== 总体准确率 (Overall Accuracy): {overall_acc:.4f} ===\n")
    
    # 3. 打印每个类别的 Precision, Recall, F1-score
    # digits=4 表示保留4位小数
    print("=== 各分类详细指标 (Precision, Recall, F1-score) ===")
    report = classification_report(y_true, y_pred, labels=labels, digits=4)
    print(report)
    
    # 4. 计算每个类别的独立准确率 (Per-class Accuracy)
    # 对于多分类，某个类别的准确率通常指 (TP + TN) / (TP + TN + FP + FN)
    print("=== 各分类独立准确率 (Per-class Accuracy) ===")
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    for i, label in enumerate(labels):
        # 真正例 (True Positives)
        TP = cm[i, i]
        # 假正例 (False Positives) - 该列之和减去TP
        FP = cm[:, i].sum() - TP
        # 假负例 (False Negatives) - 该行之和减去TP
        FN = cm[i, :].sum() - TP
        # 真负例 (True Negatives) - 总数减去(TP+FP+FN)
        TN = cm.sum() - (TP + FP + FN)
        
        # 计算该类别的 Accuracy
        class_acc = (TP + TN) / (TP + TN + FP + FN)
        print(f"类别 {label} - Accuracy: {class_acc:.4f}  (TP:{TP}, TN:{TN}, FP:{FP}, FN:{FN})")

if __name__ == "__main__":
    # 替换为你的真实文件路径
    file_path = 'save/c_model/fine-tuning/Qwen25/Coercion-47K/multi/results.jsonl' 
    
    # 为了演示，如果你没有文件，可以先用下面这段代码生成一个临时的测试文件
    """
    with open('your_data.jsonl', 'w') as f:
        f.write('{"dialogue_id": 0, "true_multi": 4, "pred_multi": 3, "model_response": "3"}\n')
        f.write('{"dialogue_id": 1, "true_multi": 4, "pred_multi": 4, "model_response": "4"}\n')
        f.write('{"dialogue_id": 2, "true_multi": 1, "pred_multi": 1, "model_response": "1"}\n')
        f.write('{"dialogue_id": 3, "true_multi": 0, "pred_multi": 2, "model_response": "2"}\n')
    """
    
    evaluate_predictions(file_path)