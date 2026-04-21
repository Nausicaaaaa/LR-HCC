# 开发可解释AI模型识别高风险背景下肝脏局灶性病变的良恶性并精准诊断HCC

1. **检测模型**：输入CT四期图像，基于nnU-Net的病灶检测模型 识别CT图像中的可疑病灶
2. **诊断模型**：
    - 诊断模型1：（Transformer端-端 + 概念激活向量解释）→ 【LR1/2，LR3，LR4，LR5，LRM】和【良性，恶性非HCC，HCC】
        - 选择准确率超过0.8 的特征为最佳特征为病灶分类结果提供归因解释
    - 诊断模型2：模型1+模型1输出的LR等级→输出：良性，恶性非HCC，HCC
    - 诊断模型3：模型2+有意义的临床变量→输出：良性，恶性非HCC，HCC

## 数据说明

|           | HCC | 恶性非HCC | 良性 | ALL | 备注                     |
| --------- | --- | --------- | ---- | --- | ------------------------ |
| 附二      | 417 | 42        | 51   | 510 | 存在数据缺失7例          |
| 西南      | 575 | 102       | 34   | 711 | 存在数据缺失2例          |
| all       | 991 | 143       | 85   |     |                          |
| 4模态数据 | 917 | 128       | 66   |1111| 去除了模态匹配不佳的数据 |

## 1. 检测模型

### 1.1 数据处理

1.`preprocess.py`进行预处理
- 文件处理：
  - 遍历地址 把四期CT（A、D、P、S or PS）对应为四种模态，命名为case_xxx_xxxx
  - 处理后的图像、标签分别保存到 `nnUNet_raw\Dataset001_HCC\imagesTr`及`labelsTr`
  - `case_name_mapping.txt`原始ID对应的 case_xxx 和类别
  - `case_skip_data.txt` 为存在缺失数据跳过的ID
- 图像处理：
  - 4 个模态的图像对齐到 A 模态的空间网格，统一shape / spacing / orientation / affine ，保证物理尺度一致
  - <MARK>注意：4个模态因为是不同时期扫的 因此不是天然的完全配准
- 标签处理：
  - 把HCC、恶性非HCC、良性三类的标签映射为不同的值（HCC=1，onHCC=2、laingxing=3）命名为case_xxx
  - <MARK>为确保标签的有效性 取4个标签的并集
  - 去除并集为0或小于500的的case 并返回case列表
  
  > 取并集的问题：有些地方不是ROI而被划分进去 导致干扰
  > 取交集的问题：更谨慎 但有些ROI会被去除掉 只能去除一些交集后ROI体素过小的case（代表模态对齐效果差）
---
- `Dataset001_HCC`存放4个模态的数据
- `Dataset002_HCC`仅AP模态数据

**可视化**

`visualization_original.py` 原始图像&标签可视化 

`visualisation.py` 处理后的case 图像&标签可视化 **检查是否正确匹配**


**数据上传服务器 ssh+rsync**
``` bash
rsync -avP --partial --append-verify --ignore-existing \
-e "ssh -p 21133 -T -c aes128-gcm@openssh.com -o Compression=no" \
/mnt/c/Users/75267/Documents/python项目/LR-HCC/data/nnUNet_raw/Dataset001_HCC/labelsTr/ \
KASR@106.120.24.118:/mnt/data/KASR/Dengsiyi/LR-HCC/data/nnUNet_raw/Dataset001_HCC/labelsTr/
```

### 1.2 nnunetv2模型训练

`python main.py`

> [!note] 小目标 ROI 分割改进方案
> 当前问题:前景 ROI 占比极小（CT 中大部分为背景）, 标准 nnUNet 配置对小目标不够敏感
> 改进方案:
>  1. 自定义 Trainer: `nnUNetTrainer_smallROI`
>       - **前景过采样**: `oversample_foreground_percent = 0.5`（原 0.0）
>       - 确保训练时 50% 的 patch 包含前景
>       - 解决小目标采样不足问题
>  2. 自定义损失函数: `FocalDiceLoss`
>       - Focal Loss 解决类别不平衡（`gamma=2.0, alpha=0.25`）
>       - 更关注难样本和小目标
>       - batch_dice=False 对每个样本单独计算 Dice
>  3. 其他
>       - 减小旋转角度**: 从 ±30° 减小到 ±15°,避免小目标被旋转出 patch

<table>
  <tr>
    <th>train loss</th>
    <th>train dice</th>
  </tr>
  <tr>
    <td><img src="visualization_output/loss_training_log_2026_4_15_12_56_06.png" height="200" /></td>
    <td><img src="visualization_output/dice_training_log_2026_4_15_12_56_06.png" height="200" /></td>
  </tr>
</table>

### 1.3 nnunetv2模型验证
|检测精度（acc）|检测平均dice|
|--|--|
|0.85|0.75|

结果样例：

<img src="visualization_output/val_comparison/case_001/case_001-slice-043.png" height="250" >


