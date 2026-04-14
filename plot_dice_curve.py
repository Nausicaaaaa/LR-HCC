#!/usr/bin/env python
"""
绘制nnUNet训练日志中的Dice曲线

用法:
    python plot_dice_curve.py --path <训练日志路径>

示例:
    python plot_dice_curve.py --path data/nnUNet_results/Dataset001_HCC-交集/nnUNetTrainer_smallROI__nnUNetPlans__2d/fold_0/training_log_2026_4_13_14_38_07.txt
"""

import argparse
import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def parse_training_log(log_path):
    """
    解析训练日志，提取epoch和dice值
    
    参数:
        log_path: 训练日志文件路径
        
    返回:
        epochs: epoch列表
        dice_values: dice值列表
    """
    epochs = []
    dice_values = []
    
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            # 匹配epoch行
            epoch_match = re.search(r'Epoch\s+(\d+)', line)
            if epoch_match:
                current_epoch = int(epoch_match.group(1))
            
            # 匹配Pseudo dice行
            dice_match = re.search(r'Pseudo dice\s+\[([\d.]+)\]', line)
            if dice_match and 'current_epoch' in locals():
                dice_value = float(dice_match.group(1))
                epochs.append(current_epoch)
                dice_values.append(dice_value)
    
    return np.array(epochs), np.array(dice_values)


def smooth_curve(values, weight=0.6):
    """
    使用指数移动平均平滑曲线
    
    参数:
        values: 原始数值数组
        weight: 平滑权重，越大平滑效果越强
        
    返回:
        smoothed: 平滑后的数值数组
    """
    smoothed = np.zeros_like(values)
    smoothed[0] = values[0]
    
    for i in range(1, len(values)):
        smoothed[i] = weight * smoothed[i-1] + (1 - weight) * values[i]
    
    return smoothed


def plot_dice_curve(epochs, dice_values, output_path=None, log_path=None):
    """
    绘制Dice曲线
    
    参数:
        epochs: epoch数组
        dice_values: dice值数组
        output_path: 输出图片路径
        log_path: 原始日志路径（用于生成默认输出路径）
    """
    # 计算平滑曲线
    smoothed_dice = smooth_curve(dice_values, weight=0.95)
    
    # 创建图形 (1:1 比例)
    plt.figure(figsize=(12, 8))
    
    # 绘制实际数值（浅色）
    plt.plot(epochs, dice_values, 'o-', color='#87CEEB', alpha=0.5, 
             markersize=3, linewidth=1, label='Actual Dice')
    
    # 绘制平滑曲线（深色）
    plt.plot(epochs, smoothed_dice, '-', color='#1E90FF', 
             linewidth=2.5, label='Smoothed Dice')
    
    # 设置标签和标题
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Dice Score', fontsize=12)
    plt.title('Training Dice Curve', fontsize=14, fontweight='bold')
    
    # 设置网格
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # 设置图例
    plt.legend(loc='lower right', fontsize=10)
    
    # 设置y轴范围
    plt.ylim(-0.05, 1.05)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存或显示
    if output_path is None and log_path is not None:
        # 从日志文件名提取时间戳，生成输出文件名
        log_name = Path(log_path).stem  # 例如: training_log_2026_4_13_14_38_07
        output_filename = f"dice_{log_name}.png"
        
        # 默认输出到 visualization_output 目录
        output_dir = Path('visualization_output')
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / output_filename
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Dice曲线已保存到: {output_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='绘制nnUNet训练日志的Dice曲线',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python plot_dice_curve.py --path training_log.txt
  python plot_dice_curve.py --path training_log.txt --output dice.png
        """
    )
    
    parser.add_argument(
        '--path',
        type=str,
        required=True,
        help='训练日志文件路径'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='输出图片路径（可选，默认保存到日志所在目录）'
    )
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    log_path = Path(args.path)
    if not log_path.exists():
        print(f"错误: 文件不存在: {log_path}")
        return
    
    # 解析日志
    print(f"正在解析日志: {log_path}")
    epochs, dice_values = parse_training_log(log_path)
    
    if len(epochs) == 0:
        print("错误: 未能从日志中解析到Dice数据")
        return
    
    print(f"成功解析 {len(epochs)} 个epoch的Dice数据")
    print(f"Dice范围: [{dice_values.min():.4f}, {dice_values.max():.4f}]")
    
    # 绘制曲线
    plot_dice_curve(epochs, dice_values, args.output, str(log_path))


if __name__ == '__main__':
    main()
