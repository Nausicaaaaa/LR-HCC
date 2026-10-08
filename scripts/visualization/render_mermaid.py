"""用matplotlib绘制3个模型的流程图并保存为PNG"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib.font_manager import FontProperties
import numpy as np
import os

# 输出到项目根目录下的 mermaid_images（脚本位于 scripts/visualization/ 下，向上回溯到根）
output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "mermaid_images")
os.makedirs(output_dir, exist_ok=True)

# 直接加载中文字体文件
FONT_PATH = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
FONT_BOLD_PATH = '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'
_fp = FontProperties(fname=FONT_PATH)
_fp_bold = FontProperties(fname=FONT_BOLD_PATH)
_fp_title = FontProperties(fname=FONT_BOLD_PATH, size=13)

def draw_box(ax, x, y, w, h, text, color='#E8F4FD', edge='#4A90D9', fontsize=8, bold=False):
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                         boxstyle="round,pad=0.05",
                         facecolor=color, edgecolor=edge, linewidth=1.2)
    ax.add_patch(box)
    fp = _fp_bold if bold else _fp
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
            fontproperties=fp, zorder=5)

def draw_diamond(ax, x, y, w, h, text, color='#FFF3CD', edge='#E6A817', fontsize=8):
    diamond = plt.Polygon([(x, y+h/2), (x+w/2, y), (x, y-h/2), (x-w/2, y)],
                          facecolor=color, edgecolor=edge, linewidth=1.2, zorder=3)
    ax.add_patch(diamond)
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
            fontproperties=_fp, zorder=5)

def draw_arrow(ax, x1, y1, x2, y2, color='#666666'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.3), zorder=2)

def draw_line_arrow(ax, points, color='#666666'):
    """绘制折线箭头, points = [(x1,y1), (x2,y2), ..., (xn,yn)]"""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    ax.plot(xs[:-1], ys[:-1], color=color, lw=1.3, zorder=2)
    ax.annotate('', xy=points[-1], xytext=points[-2],
                arrowprops=dict(arrowstyle='->', color=color, lw=1.3), zorder=2)


# ============================================================
# Model 1: DL+征象 -> LR等级
# ============================================================
def draw_model1():
    fig, ax = plt.subplots(1, 1, figsize=(13, 5))
    ax.set_xlim(-0.5, 13)
    ax.set_ylim(-0.5, 5)
    ax.axis('off')

    bw, bh = 1.6, 0.6
    sbw = 1.9

    # 主流 - 上行
    y_main = 3.5
    draw_box(ax, 0.8, y_main, bw, bh, 'Input\nCT image', '#D4EDDA', '#28A745', 8)
    draw_box(ax, 3.0, y_main, bw, bh, 'UniFormer\nBackbone', '#E8F4FD', '#4A90D9', 8)
    draw_box(ax, 5.2, y_main, bw, bh, '3D Feature\nMap', '#E8F4FD', '#4A90D9', 8)
    draw_box(ax, 7.5, y_main, bw, bh, 'GAP\n(B, 512)', '#E8F4FD', '#4A90D9', 8)

    # 分支: 征象FC - 下行
    y_branch = 1.3
    draw_box(ax, 7.5, y_branch, sbw, bh, '征象 FC\nLinear(512→512→25)', '#F8D7DA', '#DC3545', 7.5)
    draw_box(ax, 10.0, y_branch, sbw, bh, '25 征象概率\nSigmoid', '#F8D7DA', '#DC3545', 7.5)

    # Concat
    draw_diamond(ax, 7.5, 2.4, 1.1, 0.55, 'Concat', fontsize=8)

    # 输出流水线
    draw_box(ax, 10.0, 2.4, sbw, bh, '中间 FC\nLinear(537→512)\nReLU→Dropout', '#E8F4FD', '#4A90D9', 7)
    draw_box(ax, 10.0, 3.9, sbw, bh, '分类头\nLinear(512→5)', '#CCE5FF', '#004085', 8)
    draw_box(ax, 12.0, 3.9, 1.3, bh, 'LR等级\n输出', '#D4EDDA', '#28A745', 8, bold=True)

    # 箭头 - 主流
    draw_arrow(ax, 1.6, y_main, 2.2, y_main)
    draw_arrow(ax, 3.8, y_main, 4.4, y_main)
    draw_arrow(ax, 6.0, y_main, 6.7, y_main)

    # GAP → 征象FC (下)
    draw_arrow(ax, 7.5, y_main - bh/2, 7.5, y_branch + bh/2)
    # GAP → Concat
    draw_arrow(ax, 7.5, y_main - bh/2, 7.5, 2.4 + 0.28)

    # 征象FC → 25征象概率
    draw_arrow(ax, 8.45, y_branch, 9.05, y_branch)
    # 25征象概率 → Concat
    draw_line_arrow(ax, [(10.95, y_branch), (10.95, 2.4), (8.05, 2.4)])

    # Concat → 中间FC
    draw_arrow(ax, 8.05, 2.4, 9.05, 2.4)
    # 中间FC → 分类头
    draw_arrow(ax, 10.0, 2.4 + bh/2, 10.0, 3.9 - bh/2)
    # 分类头 → 输出
    draw_arrow(ax, 10.95, 3.9, 11.35, 3.9)

    ax.set_title('模型1: DL+征象 → LR等级', fontproperties=_fp_title, pad=10)
    plt.tight_layout()
    path = os.path.join(output_dir, "model1_architecture.png")
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {path}")


# ============================================================
# Model 2: DL+征象 -> 良恶性
# ============================================================
def draw_model2():
    fig, ax = plt.subplots(1, 1, figsize=(13, 5))
    ax.set_xlim(-0.5, 13)
    ax.set_ylim(-0.5, 5)
    ax.axis('off')

    bw, bh = 1.6, 0.6
    sbw = 1.9

    y_main = 3.5
    draw_box(ax, 0.8, y_main, bw, bh, 'Input\nCT image', '#D4EDDA', '#28A745', 8)
    draw_box(ax, 3.0, y_main, bw, bh, 'UniFormer\nBackbone', '#E8F4FD', '#4A90D9', 8)
    draw_box(ax, 5.2, y_main, bw, bh, '3D Feature\nMap', '#E8F4FD', '#4A90D9', 8)
    draw_box(ax, 7.5, y_main, bw, bh, 'GAP\n(B, 512)', '#E8F4FD', '#4A90D9', 8)

    y_branch = 1.3
    draw_box(ax, 7.5, y_branch, sbw, bh, '征象 FC\nLinear(512→512→25)', '#F8D7DA', '#DC3545', 7.5)
    draw_box(ax, 10.0, y_branch, sbw, bh, '25 征象概率\nSigmoid', '#F8D7DA', '#DC3545', 7.5)

    draw_diamond(ax, 7.5, 2.4, 1.1, 0.55, 'Concat', fontsize=8)

    draw_box(ax, 10.0, 2.4, sbw, bh, '中间 FC\nLinear(537→512)\nReLU→Dropout', '#E8F4FD', '#4A90D9', 7)
    draw_box(ax, 10.0, 3.9, sbw, bh, '分类头\nLinear(512→3)', '#CCE5FF', '#004085', 8)
    draw_box(ax, 12.0, 3.9, 1.3, bh, '良恶性\n输出', '#D4EDDA', '#28A745', 8, bold=True)

    draw_arrow(ax, 1.6, y_main, 2.2, y_main)
    draw_arrow(ax, 3.8, y_main, 4.4, y_main)
    draw_arrow(ax, 6.0, y_main, 6.7, y_main)

    draw_arrow(ax, 7.5, y_main - bh/2, 7.5, y_branch + bh/2)
    draw_arrow(ax, 7.5, y_main - bh/2, 7.5, 2.4 + 0.28)

    draw_arrow(ax, 8.45, y_branch, 9.05, y_branch)
    draw_line_arrow(ax, [(10.95, y_branch), (10.95, 2.4), (8.05, 2.4)])

    draw_arrow(ax, 8.05, 2.4, 9.05, 2.4)
    draw_arrow(ax, 10.0, 2.4 + bh/2, 10.0, 3.9 - bh/2)
    draw_arrow(ax, 10.95, 3.9, 11.35, 3.9)

    ax.set_title('模型2: DL+征象 → 良恶性', fontproperties=_fp_title, pad=10)
    plt.tight_layout()
    path = os.path.join(output_dir, "model2_architecture.png")
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {path}")


# ============================================================
# Model 3: DL+征象+LR等级+临床变量->良恶性
# ============================================================
def draw_model3():
    fig, ax = plt.subplots(1, 1, figsize=(14, 6.5))
    ax.set_xlim(-0.5, 14.5)
    ax.set_ylim(-1.5, 6.5)
    ax.axis('off')

    bw, bh = 1.6, 0.55
    sbw = 2.0

    # 临床分支 - 顶部
    y_clin = 5.5
    draw_box(ax, 0.8, y_clin, 1.4, bh, '临床变量\n10维', '#E2D5F1', '#7B2D8E', 8)
    draw_box(ax, 3.0, y_clin, 1.5, bh, 'z-score\n标准化', '#E2D5F1', '#7B2D8E', 8)
    draw_box(ax, 5.3, y_clin, 1.5, bh, 'clinical\n_scale', '#E2D5F1', '#7B2D8E', 8)

    # 主流
    y_main = 3.8
    draw_box(ax, 0.8, y_main, bw, bh, 'Input\nCT image', '#D4EDDA', '#28A745', 8)
    draw_box(ax, 3.0, y_main, bw, bh, 'UniFormer\nBackbone', '#E8F4FD', '#4A90D9', 8)
    draw_box(ax, 5.2, y_main, bw, bh, '3D Feature\nMap', '#E8F4FD', '#4A90D9', 8)
    draw_box(ax, 7.5, y_main, bw, bh, 'GAP\n(B, 512)', '#E8F4FD', '#4A90D9', 8)

    # 征象分支
    y_branch = 2.2
    draw_box(ax, 7.5, y_branch, sbw, bh, '征象 FC\nLinear(512→512→25)', '#F8D7DA', '#DC3545', 7)
    draw_box(ax, 10.0, y_branch, 1.5, bh, '25 征象概率', '#F8D7DA', '#DC3545', 7.5)

    # LR预测头
    y_lr = 0.7
    draw_box(ax, 7.5, y_lr, sbw, bh, 'LR预测头\nLinear(512→5)', '#FFF3CD', '#E6A817', 7)
    draw_box(ax, 10.0, y_lr, 1.3, bh, 'LR等级', '#FFF3CD', '#E6A817', 8)

    # Concat
    y_concat = 3.0
    draw_diamond(ax, 12.0, y_concat, 1.4, 0.65, 'Concat\n(B, 552)', fontsize=7)

    # 输出
    draw_box(ax, 12.0, 4.5, sbw, bh, '融合 FC\nLinear(552→512)\nReLU→Dropout', '#E8F4FD', '#4A90D9', 7)
    draw_box(ax, 12.0, 5.7, sbw, bh, '分类头\nLinear(512→3)', '#CCE5FF', '#004085', 8)
    draw_box(ax, 14.0, 5.7, 1.1, bh, '良恶性\n输出', '#D4EDDA', '#28A745', 8, bold=True)

    # 箭头 - 主流
    draw_arrow(ax, 1.6, y_main, 2.2, y_main)
    draw_arrow(ax, 3.8, y_main, 4.4, y_main)
    draw_arrow(ax, 6.0, y_main, 6.7, y_main)

    # 临床分支
    draw_arrow(ax, 1.5, y_clin, 2.25, y_clin)
    draw_arrow(ax, 3.75, y_clin, 4.55, y_clin)
    # clinical_scale → Concat
    draw_line_arrow(ax, [(6.05, y_clin), (12.0, y_clin), (12.0, y_concat + 0.33)])

    # GAP → 征象FC
    draw_arrow(ax, 7.5, y_main - bh/2, 7.5, y_branch + bh/2)
    # GAP → LR预测头
    ax.plot([7.5, 7.5], [y_main - bh/2, y_lr + bh/2], color='#666666', lw=1.3, zorder=2)
    ax.annotate('', xy=(7.5, y_lr + bh/2), xytext=(7.5, y_lr + bh/2 + 0.1),
                arrowprops=dict(arrowstyle='->', color='#666666', lw=1.3), zorder=2)

    # GAP → Concat (右侧直达)
    draw_line_arrow(ax, [(8.3, y_main), (12.0, y_main), (12.0, y_concat + 0.33)])

    # 征象FC → 25征象概率
    draw_arrow(ax, 8.5, y_branch, 9.25, y_branch)
    # 25征象概率 → Concat
    draw_line_arrow(ax, [(10.75, y_branch), (12.0, y_branch), (12.0, y_concat - 0.33)])

    # LR预测头 → LR等级
    draw_arrow(ax, 8.5, y_lr, 9.35, y_lr)
    # LR等级 → Concat
    draw_line_arrow(ax, [(10.65, y_lr), (12.0, y_lr), (12.0, y_concat - 0.33)])

    # Concat → 融合FC
    draw_arrow(ax, 12.0, y_concat + 0.33, 12.0, 4.5 - bh/2)
    # 融合FC → 分类头
    draw_arrow(ax, 12.0, 4.5 + bh/2, 12.0, 5.7 - bh/2)
    # 分类头 → 输出
    draw_arrow(ax, 13.0, 5.7, 13.45, 5.7)

    ax.set_title('模型3: DL+征象+LR等级+临床变量 → 良恶性', fontproperties=_fp_title, pad=10)
    plt.tight_layout()
    path = os.path.join(output_dir, "model3_architecture.png")
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {path}")


if __name__ == '__main__':
    print(f"Using font: {FONT_PATH}")
    draw_model1()
    draw_model2()
    draw_model3()
    print("\nAll done! Images saved to:", output_dir)
