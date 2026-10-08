import os
import random
from collections import defaultdict

# 配置路径
labels_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'labels')
all_file = os.path.join(labels_dir, 'All.txt')
output_dir = labels_dir

# 读取 ALL 文件并按 LR 等级分组
stratified_data = defaultdict(list)
first_line = True  # 标记是否为第一行（表头）

with open(all_file, 'r') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        
        # 跳过表头
        if first_line:
            first_line = False
            continue
        
        parts = line.split('\t')
        if len(parts) < 3:
            continue
        
        case_name = parts[0]
        lr_level = parts[2].strip()  # 第三列是 LR 等级
        
        stratified_data[lr_level].append(line)

print(f"发现 {len(stratified_data)} 个 LR 等级类别")
for level, cases in stratified_data.items():
    print(f"  {level}: {len(cases)} 个样本")

# 对每个类别进行 5-fold 划分
folds_train = {i: [] for i in range(1, 6)}
folds_val = {i: [] for i in range(1, 6)}

for level, cases in stratified_data.items():
    # 打乱顺序
    random.shuffle(cases)
    
    n_cases = len(cases)
    # 计算每折的验证集大小 (20%)
    val_size = max(1, int(n_cases * 0.2))
    
    # 简单循环分配：每个样本轮流作为不同 fold 的验证集一部分
    # 这里采用更标准的做法：将数据分成5份，每份轮流做验证集
    fold_size = n_cases // 5
    remainder = n_cases % 5
    
    start_idx = 0
    for fold_id in range(1, 6):
        # 当前 fold 的验证集范围
        current_fold_size = fold_size + (1 if fold_id <= remainder else 0)
        end_idx = start_idx + current_fold_size
        
        val_cases = cases[start_idx:end_idx]
        train_cases = cases[:start_idx] + cases[end_idx:]
        
        folds_val[fold_id].extend(val_cases)
        folds_train[fold_id].extend(train_cases)
        
        start_idx = end_idx

# 输出结果
for fold_id in range(1, 6):
    # 再次打乱训练集和验证集的顺序（因为是从不同类别拼接的）
    random.shuffle(folds_train[fold_id])
    random.shuffle(folds_val[fold_id])
    
    train_file = os.path.join(output_dir, f'train_fold{fold_id}.txt')
    val_file = os.path.join(output_dir, f'val_fold{fold_id}.txt')
    
    with open(train_file, 'w') as f:
        f.write('\n'.join(folds_train[fold_id]) + '\n')
        
    with open(val_file, 'w') as f:
        f.write('\n'.join(folds_val[fold_id]) + '\n')
    
    # 统计 LR 分布
    train_dist = defaultdict(int)
    val_dist = defaultdict(int)
    for line in folds_train[fold_id]:
        lr = line.split('\t')[2].strip()
        train_dist[lr] += 1
    for line in folds_val[fold_id]:
        lr = line.split('\t')[2].strip()
        val_dist[lr] += 1
    
    print(f"Fold {fold_id}: Train={len(folds_train[fold_id])}, Val={len(folds_val[fold_id])}")
    print(f"  Train LR分布: {dict(train_dist)}")
    print(f"  Val   LR分布: {dict(val_dist)}")

print("5-fold 交叉验证划分完成！")
