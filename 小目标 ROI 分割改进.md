# 小目标 ROI 分割改进方案

## 当前问题

前景 ROI 占比极小（CT 中大部分为背景）, 标准 nnUNet 配置对小目标不够敏感

---

## 改进方案

### 1. 自定义 Trainer: `nnUNetTrainer_smallROI`

**文件**: `nnunetv2/training/nnUNetTrainer/nnUNetTrainer_smallROI.py`

**主要改进**：
- **前景过采样**: `oversample_foreground_percent = 0.5`（原 0.0）
  - 确保训练时 50% 的 patch 包含前景
  - 解决小目标采样不足问题
  
- **Focal Loss + Dice Loss 组合**: 
  - Focal Loss 解决类别不平衡（`gamma=2.0, alpha=0.25`）
  - 更关注难样本和小目标
  
- **batch_dice=False**: 对每个样本单独计算 Dice
  - 对小目标更敏感
  
- **减小旋转角度**: 从 ±30° 减小到 ±15°
  - 避免小目标被旋转出 patch

### 2. 自定义损失函数: `FocalDiceLoss`

**文件**: `nnunetv2/training/loss/focal_loss.py`

Focal Loss 公式：
```
FL(pt) = -α * (1 - pt)^γ * log(pt)
```
- `pt`: 模型对正确类别的预测概率
- `γ=2.0`: 聚焦参数，越大越关注难样本
- `α=0.25`: 平衡参数，增加正样本权重

### 3. 优化预测脚本: `predict_smallROI.py`

**后处理优化**：
- **TTA (Test Time Augmentation)**: 翻转图像后平均结果
- **连通域过滤**: 移除小于阈值（默认 30 体素）的噪声
- **孔洞填充**: 保持小目标完整性
- **减小步长**: `step_size=0.5`（默认 0.5，可尝试更小）

---

## 使用步骤

### 步骤 1: 重新训练模型

```bash
# 使用优化后的 Trainer 训练
bash train_smallROI.sh

# 或单独训练某个 fold
nnUNetv2_train Dataset001_HCC 2d 0 -tr nnUNetTrainer_smallROI
```

### 步骤 2: 交叉验证

```bash
nnUNetv2_find_best_configuration Dataset001_HCC -tr nnUNetTrainer_smallROI -c 2d
```

### 步骤 3: 优化预测

```bash
# 基础预测
python predict_smallROI.py \
    -i /path/to/input \
    -o /path/to/output \
    -d Dataset001_HCC

# 启用 TTA 和后处理（推荐）
python predict_smallROI.py \
    -i /path/to/input \
    -o /path/to/output \
    -d Dataset001_HCC \
    --tta \
    --min_component_size 30 \
    --fill_holes \
    --step_size 0.5
```

---

## 其他建议

### 1. 数据层面

- **检查标注质量**: 小目标对标注误差非常敏感
- **数据增强**: 考虑添加随机裁剪（确保包含 ROI）
- **类别平衡**: 如果某些类别样本极少，考虑合并或数据合成

### 2. 模型层面

- **尝试 3D 配置**: 如果 ROI 在 z 轴有连续性，3D 可能更好
  ```bash
  nnUNetv2_train Dataset001_HCC 3d_fullres 0 -tr nnUNetTrainer_smallROI
  ```

- **调整 Patch Size**: 如果 ROI 非常小，可能需要减小 patch size
  ```bash
  # 编辑 plans.json 或使用自定义 Planner
  ```

### 3. 损失函数调参

如果效果不佳，可以尝试调整 Focal Loss 参数：
- `gamma=1.5`: 减少对难样本的过度关注
- `gamma=3.0`: 更关注难样本
- `alpha=0.5`: 平衡正负样本权重

### 4. 后处理调参

根据 ROI 实际大小调整：
- `--min_component_size`: 根据最小 ROI 大小设置（如 20-50）
- `--step_size 0.25`: 更精细的滑动窗口（更慢但更准）

---
