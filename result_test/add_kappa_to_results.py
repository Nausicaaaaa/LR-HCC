import pandas as pd
import numpy as np

# 读取CSV文件
df = pd.read_csv('/mnt/data/KASR/Dengsiyi/LR-HCC/result_test/外部测试结果.csv')

# 计算Kappa值
def calculate_kappa(row):
    tp = row['tp']
    fp = row['fp']
    tn = row['tn']
    fn = row['fn']
    
    # 总样本数
    total = tp + fp + tn + fn
    
    if total == 0:
        return 0.0
    
    # 观测一致率 (Po)
    po = (tp + tn) / total
    
    # 期望一致率 (Pe)
    # 预测为正的总数
    pred_pos = tp + fp
    # 预测为负的总数
    pred_neg = tn + fn
    # 实际为正的总数
    actual_pos = tp + fn
    # 实际为负的总数
    actual_neg = tn + fn
    
    # Pe = (预测为正的概率 * 实际为正的概率) + (预测为负的概率 * 实际为负的概率)
    pe = ((pred_pos / total) * (actual_pos / total)) + ((pred_neg / total) * (actual_neg / total))
    
    # Kappa = (Po - Pe) / (1 - Pe)
    if pe == 1.0:
        return 0.0
    
    kappa = (po - pe) / (1 - pe)
    
    return kappa

# 计算每行的Kappa值
kappa_values = df.apply(calculate_kappa, axis=1)

# 在precision列后插入kappa列
precision_idx = df.columns.get_loc('precision')
df.insert(precision_idx + 1, 'kappa', kappa_values)

# 保存结果
df.to_csv('/mnt/data/KASR/Dengsiyi/LR-HCC/result_test/外部测试结果.csv', index=False)

print("Kappa值已计算并添加到CSV文件中")
print(f"Kappa值范围: {kappa_values.min():.4f} ~ {kappa_values.max():.4f}")
print(f"平均Kappa值: {kappa_values.mean():.4f}")
