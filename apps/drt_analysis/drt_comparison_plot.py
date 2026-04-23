"""
DRT对比绘图脚本
在一张图中同时绘制两个或多个EIS文件的DRT-2D图像进行对比
"""
import sys
sys.path.append("E:/PycharmProjects/PythonProject1")
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import glob
import os
import re
import warnings
warnings.filterwarnings('ignore')

# 导入核心库
from battery_analysis.drt import DRTAnalyzerFinal, read_and_preprocess_eis

# 字体和负号显示设置
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.default'] = 'regular'

# 自定义刻度格式化函数，将 Unicode 负号替换为普通减号
from matplotlib.ticker import FuncFormatter

def fix_minus_tex_formatter(x, pos):
    """使用 LaTeX 格式但避免 Unicode 负号"""
    if x == 0:
        return r'$0$'
    if x >= 1:
        return f'${x:.0f}$'
    exponent = int(np.floor(np.log10(abs(x))))
    return f'$10^{{{exponent}}}$'


def plot_drt_comparison(file_path1, file_path2, label1=None, label2=None, save_path=None):
    """
    在一张图中对比绘制两个EIS文件的DRT结果

    Args:
        file_path1: 第一个EIS文件路径
        file_path2: 第二个EIS文件路径
        label1: 第一条曲线的标签（默认使用文件名）
        label2: 第二条曲线的标签（默认使用文件名）
        save_path: 保存路径（可选）

    Returns:
        fig: matplotlib figure对象
    """
    # 生成默认标签
    if label1 is None:
        label1 = file_path1.split('\\')[-1].replace('.txt', '')
    if label2 is None:
        label2 = file_path2.split('\\')[-1].replace('.txt', '')

    print("="*70)
    print("DRT Comparison Analysis")
    print("="*70)

    # 读取并处理第一个文件
    print(f"\n处理文件1: {file_path1.split('\\')[-1]}")
    freq1, Z_re1, Z_im1, Rs1 = read_and_preprocess_eis(file_path1)
    analyzer1 = DRTAnalyzerFinal(freq1, Z_re1, Z_im1, n_tau=200)
    analyzer1.fit_optimized(lambda_reg=None)
    print(f"文件1 - 最优lambda: {analyzer1.lambda_opt:.3e}")

    # 读取并处理第二个文件
    print(f"\n处理文件2: {file_path2.split('\\')[-1]}")
    freq2, Z_re2, Z_im2, Rs2 = read_and_preprocess_eis(file_path2)
    analyzer2 = DRTAnalyzerFinal(freq2, Z_re2, Z_im2, n_tau=200)
    analyzer2.fit_optimized(lambda_reg=None)
    print(f"文件2 - 最优lambda: {analyzer2.lambda_opt:.3e}")

    # 创建对比图
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.35)

    # ==================== 1. DRT谱图对比 ====================
    ax1 = fig.add_subplot(gs[0, :])  # 占据第一行全部
    ax1.semilogx(analyzer1.tau, analyzer1.gamma * 1000, 'o-',
                 label=label1, markersize=5, linewidth=2.5, color='blue', alpha=0.8)
    ax1.semilogx(analyzer2.tau, analyzer2.gamma * 1000, 's-',
                 label=label2, markersize=5, linewidth=2.5, color='red', alpha=0.8)

    ax1.fill_between(analyzer1.tau, analyzer1.gamma * 1000, alpha=0.2, color='blue')
    ax1.fill_between(analyzer2.tau, analyzer2.gamma * 1000, alpha=0.2, color='red')

    ax1.set_xlabel("Relaxation Time $\\tau$ (s)", fontsize=13, fontweight='bold')
    ax1.set_ylabel("$\\gamma$ (mOhm·cm$^{2}$)", fontsize=13, fontweight='bold')
    ax1.set_title('DRT Spectrum Comparison', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11, loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(FuncFormatter(fix_minus_tex_formatter))

    # 标注峰值
    for analyzer, label, color, offset in [(analyzer1, label1, 'blue', 10),
                                            (analyzer2, label2, 'red', -10)]:
        peaks, _ = find_peaks(analyzer.gamma, height=np.max(analyzer.gamma)*0.05)
        if len(peaks) > 0:
            sorted_peaks = peaks[np.argsort(-analyzer.gamma[peaks])][:5]
            for peak in sorted_peaks:
                ax1.annotate(f'{analyzer.tau[peak]:.2g}s',
                           xy=(analyzer.tau[peak], analyzer.gamma[peak] * 1000),
                           xytext=(offset, 15), textcoords='offset points',
                           fontsize=8, color=color,
                           bbox=dict(boxstyle='round,pad=0.3',
                                     facecolor='white', alpha=0.8, edgecolor=color),
                           arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0',
                                          color=color, lw=1))

    # ==================== 2. Nyquist图对比 ====================
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(analyzer1.Z_re * 1000, analyzer1.Z_im * 1000, 'o-',
             label=label1, markersize=6, linewidth=2, color='blue', alpha=0.8)
    ax2.plot(analyzer2.Z_re * 1000, analyzer2.Z_im * 1000, 's-',
             label=label2, markersize=6, linewidth=2, color='red', alpha=0.8)

    if analyzer1.Z_fit_re is not None:
        ax2.plot(analyzer1.Z_fit_re * 1000, analyzer1.Z_fit_im * 1000, '--',
                 label=f'{label1} (DRT)', linewidth=2, color='blue', alpha=0.5)
    if analyzer2.Z_fit_re is not None:
        ax2.plot(analyzer2.Z_fit_re * 1000, analyzer2.Z_fit_im * 1000, '--',
                 label=f'{label2} (DRT)', linewidth=2, color='red', alpha=0.5)

    ax2.set_xlabel("Z' (mOhm·cm$^{2}$)", fontsize=11, fontweight='bold')
    ax2.set_ylabel("-Z'' (mOhm·cm$^{2}$)", fontsize=11, fontweight='bold')
    ax2.set_title('Nyquist Plot Comparison', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9, loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')

    # ==================== 3. 实部对比 ====================
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.semilogx(analyzer1.freq, analyzer1.Z_re * 1000, 'o-',
                 label=f'{label1} Exp', markersize=5, linewidth=2, color='blue', alpha=0.8)
    ax3.semilogx(analyzer2.freq, analyzer2.Z_re * 1000, 's-',
                 label=f'{label2} Exp', markersize=5, linewidth=2, color='red', alpha=0.8)

    if analyzer1.Z_fit_re is not None:
        ax3.semilogx(analyzer1.freq, analyzer1.Z_fit_re * 1000, '--',
                    label=f'{label1} Fit', linewidth=1.5, color='darkblue', alpha=0.6)
    if analyzer2.Z_fit_re is not None:
        ax3.semilogx(analyzer2.freq, analyzer2.Z_fit_re * 1000, '--',
                    label=f'{label2} Fit', linewidth=1.5, color='darkred', alpha=0.6)

    ax3.set_xlabel("Frequency (Hz)", fontsize=11, fontweight='bold')
    ax3.set_ylabel("Z' (mOhm·cm$^{2}$)", fontsize=11, fontweight='bold')
    ax3.set_title('Real Part Comparison', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9, loc='best')
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(FuncFormatter(fix_minus_tex_formatter))

    # ==================== 4. 虚部对比 ====================
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.semilogx(analyzer1.freq, analyzer1.Z_im * 1000, 'o-',
                 label=f'{label1} Exp', markersize=5, linewidth=2, color='blue', alpha=0.8)
    ax4.semilogx(analyzer2.freq, analyzer2.Z_im * 1000, 's-',
                 label=f'{label2} Exp', markersize=5, linewidth=2, color='red', alpha=0.8)

    if analyzer1.Z_fit_im is not None:
        ax4.semilogx(analyzer1.freq, analyzer1.Z_fit_im * 1000, '--',
                    label=f'{label1} Fit', linewidth=1.5, color='darkblue', alpha=0.6)
    if analyzer2.Z_fit_im is not None:
        ax4.semilogx(analyzer2.freq, analyzer2.Z_fit_im * 1000, '--',
                    label=f'{label2} Fit', linewidth=1.5, color='darkred', alpha=0.6)

    ax4.set_xlabel("Frequency (Hz)", fontsize=11, fontweight='bold')
    ax4.set_ylabel("-Z'' (mOhm·cm$^{2}$)", fontsize=11, fontweight='bold')
    ax4.set_title('Imaginary Part Comparison', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9, loc='best')
    ax4.grid(True, alpha=0.3)
    ax4.xaxis.set_major_formatter(FuncFormatter(fix_minus_tex_formatter))

    # 总标题
    fig.suptitle('DRT Comparison: ' + label1 + ' vs ' + label2,
                 fontsize=16, fontweight='bold', y=0.995)

    # 保存图像
    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n对比图已保存至: {save_path}")

    # 打印统计信息
    print("\n" + "="*70)
    print("Fitting Statistics Comparison")
    print("="*70)
    print(f"\n{label1}:")
    print(f"  Rs: {Rs1*1000:.3f} mOhm·cm²")
    print(f"  RMSE: {analyzer1.calculate_rmse()*1000:.4f} mOhm·cm²")
    print(f"  Lambda: {analyzer1.lambda_opt:.3e}")

    print(f"\n{label2}:")
    print(f"  Rs: {Rs2*1000:.3f} mOhm·cm²")
    print(f"  RMSE: {analyzer2.calculate_rmse()*1000:.4f} mOhm·cm²")
    print(f"  Lambda: {analyzer2.lambda_opt:.3e}")

    print("\n" + "="*70)

    return fig


def plot_drt_batch(folder_path, pattern='*75*1h*.txt', label_pattern=None, save_path=None):
    """
    批量绘制多个EIS文件的DRT对比图

    Args:
        folder_path: 文件夹路径
        pattern: 文件匹配模式，默认 '*75*1h*.txt'
        label_pattern: 标签提取模式（可选），例如 '1h-(\\d+)A' 提取电流值
        save_path: 保存路径（可选）

    Returns:
        fig: matplotlib figure对象
    """
    # 查找匹配的文件
    search_pattern = os.path.join(folder_path, pattern)
    files = sorted(glob.glob(search_pattern))

    if not files:
        print(f"未找到匹配的文件: {search_pattern}")
        return None

    if len(files) < 2:
        print(f"至少需要2个文件进行对比，当前只找到 {len(files)} 个")
        return None

    print("="*70)
    print("DRT Batch Comparison Analysis")
    print("="*70)
    print(f"\n找到 {len(files)} 个文件:")
    for f in files:
        print(f"  - {os.path.basename(f)}")

    # 处理所有文件
    analyzers = []
    labels = []

    for file_path in files:
        print(f"\n处理文件: {os.path.basename(file_path)}")
        try:
            freq, Z_re, Z_im, Rs = read_and_preprocess_eis(file_path)
            analyzer = DRTAnalyzerFinal(freq, Z_re, Z_im, n_tau=200)
            analyzer.fit_optimized(lambda_reg=0.3e4)
            analyzers.append(analyzer)

            # 生成标签
            basename = os.path.basename(file_path).replace('.txt', '')
            if label_pattern:
                match = re.search(label_pattern, basename)
                if match:
                    label = match.group(1) if match.groups() else basename
                else:
                    label = basename
            else:
                label = basename
            labels.append(label)

            print(f"  最优lambda: {analyzer.lambda_opt:.3e}")

        except Exception as e:
            print(f"  处理失败: {e}")
            continue

    if len(analyzers) < 2:
        print(f"成功处理的文件少于2个，无法绘制对比图")
        return None

    # 定义颜色循环
    colors = plt.cm.tab10(np.linspace(0, 1, len(analyzers)))
    markers = ['o', 's', '^', 'd', 'v', '<', '>', 'p', '*', 'h']

    # 创建对比图
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.35)

    # ==================== 1. DRT谱图对比 ====================
    ax1 = fig.add_subplot(gs[0, :])

    for i, (analyzer, label, color, marker) in enumerate(zip(analyzers, labels, colors, markers)):
        ax1.semilogx(analyzer.tau, analyzer.gamma * 1000, marker + '-',
                     label=label, markersize=5, linewidth=2.5,
                     color=color, alpha=0.8)
        ax1.fill_between(analyzer.tau, analyzer.gamma * 1000, alpha=0.15, color=color)

    ax1.set_xlabel("Relaxation Time $\\tau$ (s)", fontsize=13, fontweight='bold')
    ax1.set_ylabel("$\\gamma$ (mOhm·cm$^{2}$)", fontsize=13, fontweight='bold')
    ax1.set_title('DRT Spectrum Comparison', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10, loc='best', ncol=min(3, len(labels)))
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(FuncFormatter(fix_minus_tex_formatter))

    # 标注峰值（为每条曲线标注所有的峰）
    for analyzer, label, color, i in zip(analyzers, labels, colors, range(len(analyzers))):
        peaks, _ = find_peaks(analyzer.gamma, height=np.max(analyzer.gamma)*0.05)
        if len(peaks) > 0:
            # 根据索引设置不同的偏移方向，避免重叠
            offset = 15 if i % 2 == 0 else -15
            for peak in peaks:
                ax1.annotate(f'{analyzer.tau[peak]:.2g}s',
                           xy=(analyzer.tau[peak], analyzer.gamma[peak] * 1000),
                           xytext=(offset, 20), textcoords='offset points',
                           fontsize=7, color=color,
                           bbox=dict(boxstyle='round,pad=0.3',
                                     facecolor='white', alpha=0.8, edgecolor=color, linewidth=0.8),
                           arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0',
                                          color=color, lw=0.8))

    # ==================== 2. Nyquist图对比 ====================
    ax2 = fig.add_subplot(gs[1, 0])

    for i, (analyzer, label, color, marker) in enumerate(zip(analyzers, labels, colors, markers)):
        ax2.plot(analyzer.Z_re * 1000, analyzer.Z_im * 1000, marker + '-',
                 label=label, markersize=5, linewidth=2,
                 color=color, alpha=0.8)
        if analyzer.Z_fit_re is not None:
            ax2.plot(analyzer.Z_fit_re * 1000, analyzer.Z_fit_im * 1000, '--',
                     linewidth=1.5, color=color, alpha=0.5)

    ax2.set_xlabel("Z' (mOhm·cm$^{2}$)", fontsize=11, fontweight='bold')
    ax2.set_ylabel("-Z'' (mOhm·cm$^{2}$)", fontsize=11, fontweight='bold')
    ax2.set_title('Nyquist Plot Comparison', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9, loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')

    # ==================== 3. 实部对比 ====================
    ax3 = fig.add_subplot(gs[1, 1])

    for i, (analyzer, label, color, marker) in enumerate(zip(analyzers, labels, colors, markers)):
        ax3.semilogx(analyzer.freq, analyzer.Z_re * 1000, marker + '-',
                     label=label, markersize=4, linewidth=2,
                     color=color, alpha=0.8)
        if analyzer.Z_fit_re is not None:
            ax3.semilogx(analyzer.freq, analyzer.Z_fit_re * 1000, '--',
                         linewidth=1.2, color=color, alpha=0.5)

    ax3.set_xlabel("Frequency (Hz)", fontsize=11, fontweight='bold')
    ax3.set_ylabel("Z' (mOhm·cm$^{2}$)", fontsize=11, fontweight='bold')
    ax3.set_title('Real Part Comparison', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9, loc='best')
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(FuncFormatter(fix_minus_tex_formatter))

    # ==================== 4. 虚部对比 ====================
    ax4 = fig.add_subplot(gs[1, 2])

    for i, (analyzer, label, color, marker) in enumerate(zip(analyzers, labels, colors, markers)):
        ax4.semilogx(analyzer.freq, analyzer.Z_im * 1000, marker + '-',
                     label=label, markersize=4, linewidth=2,
                     color=color, alpha=0.8)
        if analyzer.Z_fit_im is not None:
            ax4.semilogx(analyzer.freq, analyzer.Z_fit_im * 1000, '--',
                         linewidth=1.2, color=color, alpha=0.5)

    ax4.set_xlabel("Frequency (Hz)", fontsize=11, fontweight='bold')
    ax4.set_ylabel("-Z'' (mOhm·cm$^{2}$)", fontsize=11, fontweight='bold')
    ax4.set_title('Imaginary Part Comparison', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9, loc='best')
    ax4.grid(True, alpha=0.3)
    ax4.xaxis.set_major_formatter(FuncFormatter(fix_minus_tex_formatter))

    # 总标题
    fig.suptitle(f'DRT Batch Comparison ({len(analyzers)} files)',
                 fontsize=16, fontweight='bold', y=0.995)

    # 保存图像
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n对比图已保存至: {save_path}")

    # 打印统计信息
    print("\n" + "="*70)
    print("Fitting Statistics Summary")
    print("="*70)
    for label, analyzer in zip(labels, analyzers):
        print(f"\n{label}:")
        print(f"  RMSE: {analyzer.calculate_rmse()*1000:.4f} mOhm·cm^2")
        print(f"  Lambda: {analyzer.lambda_opt:.3e}")

    print("\n" + "="*70)

    return fig


if __name__ == "__main__":
    import os

    # ==================== 使用方式选择 ====================
    # True: 批量对比模式, False: 双文件对比模式
    USE_BATCH_MODE = True

    if USE_BATCH_MODE:
        # ==================== 批量对比模式 ====================
        folder_path = r"E:\Datasets\314Ah\Bat1\RPT2\SOC"
        pattern = "*D30*.txt"  # 文件匹配模式

        # 可选：使用正则表达式提取标签（例如提取电流值）
        # '1h-(\d+)A' 会从 "#1-RPT2-D75%_1h-120A.txt" 中提取 "120A"
        label_pattern = r"1h-(\d+)A"

        save_path = r"E:\PycharmProjects\PythonProject1\output\DRT_batch_comparison.png"

        fig = plot_drt_batch(folder_path, pattern=pattern,
                            label_pattern=label_pattern,
                            save_path=None)

    else:
        # ==================== 双文件对比模式 ====================
        file1 = r"E:\Datasets\314Ah\Bat1\RPT2\#1-RPT2-D75%_1h-120A.txt"
        file2 = r"E:\Datasets\314Ah\Bat1\RPT2\#1-RPT2-D75%_1h-160A.txt"

        label1 = "120A"
        label2 = "160A"

        fig = plot_drt_comparison(file1, file2, label1=label1, label2=label2, save_path=None)

    plt.show()
