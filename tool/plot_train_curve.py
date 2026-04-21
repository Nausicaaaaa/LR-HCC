#!/usr/bin/env python
"""
绘制nnUNet训练日志中的Dice曲线和Loss曲线

用法:
    python plot_train_curve.py --path <训练日志路径>

示例:
    python plot_train_curve.py --path data/nnUNet_results/Dataset001_HCC/nnUNetTrainer_smallROI__nnUNetPlans__2d/fold_0/training_log_2026_4_15_12_56_06.txt
"""

import argparse
import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def parse_training_log(log_path):
    """
    解析训练日志，提取epoch、dice值和loss值
    
    参数:
        log_path: 训练日志文件路径
        
    返回:
        epochs: epoch列表
        dice_values: dice值列表
        train_losses: 训练loss列表
        val_losses: 验证loss列表
    """
    epochs = []
    dice_values = []
    train_losses = []
    val_losses = []
    
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
            
            # 匹配训练loss行 (格式: train_loss -0.0233 或 train loss : 0.1234)
            train_loss_match = re.search(r'train_loss\s+(-?[\d.]+)', line, re.IGNORECASE)
            if train_loss_match and 'current_epoch' in locals():
                # 确保是当前epoch的训练loss
                if len(train_losses) < len(epochs):
                    train_losses.append(float(train_loss_match.group(1)))
            
            # 匹配验证loss行 (格式: val_loss -0.1437 或 validation loss : 0.1234)
            val_loss_match = re.search(r'val_loss\s+(-?[\d.]+)', line, re.IGNORECASE)
            if val_loss_match and 'current_epoch' in locals():
                # 确保是当前epoch的验证loss
                if len(val_losses) < len(epochs):
                    val_losses.append(float(val_loss_match.group(1)))
    
    return np.array(epochs), np.array(dice_values), np.array(train_losses), np.array(val_losses)


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
    
    # 创建图形
    fig, ax = plt.figure(figsize=(12, 8)), plt.gca()
    
    # 设置背景色为白色（nnUNet v2 默认风格）
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    # 绘制实际数值（matplotlib 默认蓝色 tab:blue）
    ax.plot(epochs, dice_values, '-', color='#1f77b4', alpha=0.3, 
             linewidth=1, label='Actual Dice')
    
    # 绘制平滑曲线（matplotlib 默认蓝色 tab:blue，深色）
    ax.plot(epochs, smoothed_dice, '-', color='#1f77b4', 
             linewidth=2.5, label='Smoothed Dice')
    
    # 设置标签和标题
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Dice Score', fontsize=12)
    ax.set_title('Training Dice Curve', fontsize=14, fontweight='bold')
    
    # 设置网格（灰色虚线，nnUNet v2 默认风格）
    ax.grid(True, linestyle='--', color='gray', alpha=0.3, linewidth=0.8)
    ax.set_axisbelow(True)
    
    # 设置图例
    ax.legend(loc='lower right', fontsize=10)
    
    # 设置y轴范围
    ax.set_ylim(0, 1.0)
    
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
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Dice曲线已保存到: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_loss_curve(epochs, train_losses, val_losses, output_path=None, log_path=None):
    """
    绘制Loss曲线（训练loss和验证loss放在同一张图）
    
    参数:
        epochs: epoch数组
        train_losses: 训练loss数组
        val_losses: 验证loss数组
        output_path: 输出图片路径
        log_path: 原始日志路径（用于生成默认输出路径）
    """
    # 创建图形
    fig, ax = plt.figure(figsize=(12, 8)), plt.gca()
    
    # 设置背景色为白色（nnUNet v2 默认风格）
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    # 计算平滑曲线并绘制
    if len(train_losses) > 0:
        smoothed_train = smooth_curve(train_losses, weight=0.95)
        # 绘制训练loss实际值（matplotlib 默认蓝色 tab:blue）
        ax.plot(epochs[:len(train_losses)], train_losses, '-', color='#1f77b4', 
                 alpha=0.3, linewidth=1, label='Train Loss (Actual)')
        # 绘制训练loss平滑曲线（matplotlib 默认蓝色 tab:blue）
        ax.plot(epochs[:len(train_losses)], smoothed_train, '-', color='#1f77b4', 
                 linewidth=2.5, label='Train Loss (Smoothed)')
    
    if len(val_losses) > 0:
        smoothed_val = smooth_curve(val_losses, weight=0.95)
        # 绘制验证loss实际值（matplotlib 默认橙色 tab:orange）
        ax.plot(epochs[:len(val_losses)], val_losses, '-', color='#ff7f0e', 
                 alpha=0.3, linewidth=1, label='Val Loss (Actual)')
        # 绘制验证loss平滑曲线（matplotlib 默认橙色 tab:orange）
        ax.plot(epochs[:len(val_losses)], smoothed_val, '-', color='#ff7f0e', 
                 linewidth=2.5, label='Val Loss (Smoothed)')
    
    # 设置标签和标题
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training and Validation Loss Curve', fontsize=14, fontweight='bold')
    
    # 设置网格（灰色虚线，nnUNet v2 默认风格）
    ax.grid(True, linestyle='--', color='gray', alpha=0.3, linewidth=0.8)
    ax.set_axisbelow(True)
    
    # 设置图例
    ax.legend(loc='upper right', fontsize=9)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存或显示
    if output_path is None and log_path is not None:
        # 从日志文件名提取时间戳，生成输出文件名
        log_name = Path(log_path).stem  # 例如: training_log_2026_4_13_14_38_07
        output_filename = f"loss_{log_name}.png"
        
        # 默认输出到 visualization_output 目录
        output_dir = Path('visualization_output')
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / output_filename
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Loss曲线已保存到: {output_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='绘制nnUNet训练日志的Dice曲线和Loss曲线',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python plot_train_curve.py --path training_log.txt
  python plot_train_curve.py --path training_log.txt --output_dice dice.png --output_loss loss.png
        """
    )
    
    parser.add_argument(
        '--path',
        type=str,
        required=True,
        help='训练日志文件路径'
    )
    
    parser.add_argument(
        '--output_dice',
        type=str,
        default=None,
        help='Dice曲线输出图片路径（可选，默认保存到visualization_output目录）'
    )
    
    parser.add_argument(
        '--output_loss',
        type=str,
        default=None,
        help='Loss曲线输出图片路径（可选，默认保存到visualization_output目录）'
    )
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    log_path = Path(args.path)
    if not log_path.exists():
        print(f"错误: 文件不存在: {log_path}")
        return
    
    # 解析日志
    print(f"正在解析日志: {log_path}")
    epochs, dice_values, train_losses, val_losses = parse_training_log(log_path)
    
    if len(epochs) == 0:
        print("错误: 未能从日志中解析到Dice数据")
        return
    
    print(f"成功解析 {len(epochs)} 个epoch的数据")
    print(f"Dice范围: [{dice_values.min():.4f}, {dice_values.max():.4f}]")
    if len(train_losses) > 0:
        print(f"Train Loss范围: [{train_losses.min():.4f}, {train_losses.max():.4f}]")
    if len(val_losses) > 0:
        print(f"Validation Loss范围: [{val_losses.min():.4f}, {val_losses.max():.4f}]")
    
    # 绘制Dice曲线
    plot_dice_curve(epochs, dice_values, args.output_dice, str(log_path))
    
    # 绘制Loss曲线
    plot_loss_curve(epochs, train_losses, val_losses, args.output_loss, str(log_path))


if __name__ == '__main__':
    main()
