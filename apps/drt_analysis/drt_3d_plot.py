"""
DRT三维批量对比绘图模块
在三维空间中批量绘制多个EIS文件的DRT图像进行对比
参考DRT_comparison_plot.py的批量处理逻辑，实现3D可视化
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import find_peaks
import glob
import os
import re
import warnings
warnings.filterwarnings('ignore')
import sys
sys.path.append("E:/PycharmProjects/Paper1")
# 导入核心库
from battery_analysis.drt import DRTAnalyzerFinal, read_and_preprocess_eis

# 字体和负号显示设置
plt.rcParams['font.sans-serif'] = ['Calibri']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.default'] = 'regular'

# 论文风格设置
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['xtick.major.width'] = 1.0
plt.rcParams['ytick.major.width'] = 1.0


def plot_drt_batch_3d(folder_path, pattern='*75*1h*.txt', label_pattern=None,
                      save_path=None, csv_save_path=None, error_csv_save_path=None, view_elev=25, view_azim=60):
    """
    简化版3D对比图 - 去除填充，只显示曲线，Y轴显示SOC百分比

    Args:
        folder_path: 文件夹路径
        pattern: 文件匹配模式
        label_pattern: 标签提取模式（可选）
        save_path: 图像保存路径（可选）
        csv_save_path: 峰值数据CSV保存路径（可选）
        error_csv_save_path: 拟合误差数据CSV保存路径（可选，默认保存到output文件夹）
            包含：RMSE、MAPE、R²(实部)、R²(虚部)、lambda_opt等指标

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
    print("DRT 3D Simple Comparison")
    print("="*70)
    print(f"\n找到 {len(files)} 个文件:")
    for f in files:
        print(f"  - {os.path.basename(f)}")

    # 处理所有文件并提取SOC信息
    analyzers = []
    labels = []
    soc_values = []
    peak_data = []  # 存储每个SOC对应的最高峰峰值

    for file_path in files:
        print(f"\n处理文件: {os.path.basename(file_path)}")
        try:
            freq, Z_re, Z_im, Rs = read_and_preprocess_eis(file_path)
            analyzer = DRTAnalyzerFinal(freq, Z_re, Z_im, n_tau=200)
            analyzer.fit_optimized(lambda_reg=0.3e4)        # 自动搜索最优lambda。0.3e4可选

            # 从文件名提取SOC值（例如 #1-RPT2-D25-1h.txt -> 25）
            basename = os.path.basename(file_path)
            soc_match = re.search(r"D(\d+)", basename)  # 这里需要修改一下
            if soc_match:
                soc = float(soc_match.group(1))
            else:
                print(f"  警告：无法从文件名提取SOC值，使用默认值")
                soc = 0.0

            # 生成标签
            if label_pattern:
                match = re.search(label_pattern, basename.replace('.txt', ''))
                if match:
                    label = match.group(1) if match.groups() else basename
                else:
                    label = basename
            else:
                label = basename

            analyzers.append({'analyzer': analyzer, 'soc': soc, 'label': label})
            labels.append(label)
            soc_values.append(soc)

            print(f"  SOC: {soc}%")
            print(f"  最优lambda: {analyzer.lambda_opt:.3e}")

        except Exception as e:
            print(f"  处理失败: {e}")
            continue

    # 按SOC值排序
    sorted_data = sorted(analyzers, key=lambda x: x['soc'])
    analyzers = [item['analyzer'] for item in sorted_data]
    soc_values = [item['soc'] for item in sorted_data]
    labels = [item['label'] for item in sorted_data]

    print(f"\n排序后的SOC值: {soc_values}")

    if len(analyzers) < 2:
        print(f"成功处理的文件少于2个，无法绘制对比图")
        return None

    # 定义颜色方案 - 使用viridis渐变色方案，颜色随SOC值变化
    # 将SOC值归一化到[0, 1]区间，然后映射到颜色
    soc_min, soc_max = min(soc_values), max(soc_values)
    normalized_soc = [(soc - soc_min) / (soc_max - soc_min) if soc_max > soc_min else 0.5
                      for soc in soc_values]
    colors = plt.cm.viridis(np.array(normalized_soc))

    # 创建3D图形 - 更宽的画布
    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot(111, projection='3d')

    # 调试：检查axes类型
    print(f"Axes type: {type(ax)}")
    print(f"Has 3D projection: {hasattr(ax, 'zaxis')}")

    # 计算X轴对数范围（需要在绘制循环之前计算）
    tau_min = min([analyzer.tau.min() for analyzer in analyzers])
    tau_max = max([analyzer.tau.max() for analyzer in analyzers])
    log_min = int(np.floor(np.log10(tau_min)))
    log_max = int(np.ceil(np.log10(tau_max)))

    # 用于存储垂足点的坐标
    foot_points_x = []
    foot_points_y = []
    foot_points_z = []
    foot_points_soc = []

    # 绘制每个DRT谱 - 只绘制曲线
    for idx, (analyzer, soc, label, color) in enumerate(zip(analyzers, soc_values, labels, colors)):
        # 只绘制非零值
        mask = analyzer.gamma > 0
        tau_valid = analyzer.tau[mask]
        gamma_valid = analyzer.gamma[mask]

        if len(tau_valid) == 0:
            continue

        # 创建Y轴位置 - 使用SOC值
        y_pos = np.ones_like(tau_valid) * soc

        # 绘制DRT曲线 - 更粗的线条
        ax.plot(np.log10(tau_valid), y_pos, gamma_valid * 1000,
                linewidth=4,
                label=label,
                color=color,
                alpha=1.0,
                zorder=10)

        # 添加从最高峰值点出发，垂直于YZ面的垂线和标记
        peaks, _ = find_peaks(gamma_valid, height=np.max(gamma_valid)*0.05)
        if len(peaks) > 0:
            # 找到最高峰值点
            max_peak_idx = peaks[np.argmax(gamma_valid[peaks])]
            x_peak = np.log10(tau_valid[max_peak_idx])
            y_peak = soc
            z_peak = gamma_valid[max_peak_idx] * 1000

            print(f"{soc} 峰值大小: {z_peak:.6f}")

            # 保存峰值数据
            peak_data.append({
                'SOC': soc,
                'peak_gamma_mOhm_cm2': z_peak,
                'tau_peak_s': tau_valid[max_peak_idx],
                'log_tau_peak': x_peak
            })

            # 画垂线到YZ平面（x=log_max）
            ax.plot([log_max, x_peak],
                    [y_peak, y_peak],
                    [z_peak, z_peak],
                    color=color,
                    linestyle='--',
                    linewidth=2,
                    alpha=0.8,
                    zorder=15)

            # 在垂足处添加标记
            ax.scatter(log_max, y_peak, z_peak,
                      color=color,
                      s=60,
                      marker='o',  # 圆形标记
                      edgecolors='black',
                      linewidths=2,
                      zorder=25)

            # 在最高峰处添加标记
            ax.scatter(x_peak, y_peak, z_peak,
                      color=color,
                      s=120,
                      marker='o',
                      edgecolors='black',
                      linewidths=2,
                      zorder=20)

            # 画垂线到XY平面（垂直于XY面，向下到z=0）
            ax.plot([x_peak, x_peak],
                    [y_peak, y_peak],
                    [z_peak, 0],
                    color=color,
                    linestyle='--',
                    linewidth=2,
                    alpha=0.6,
                    zorder=15)

            # 存储垂足点坐标
            foot_points_x.append(log_max)
            foot_points_y.append(y_peak)
            foot_points_z.append(z_peak)
            foot_points_soc.append(soc)

    # 连接所有垂足点（按SOC排序）
    if len(foot_points_x) > 1:
        # 按SOC值排序
        sorted_indices = np.argsort(foot_points_soc)
        foot_points_x = [foot_points_x[i] for i in sorted_indices]
        foot_points_y = [foot_points_y[i] for i in sorted_indices]
        foot_points_z = [foot_points_z[i] for i in sorted_indices]

        # 使用曲线连接垂足点（三次样条插值）
        from scipy.interpolate import CubicSpline
        cs_y = CubicSpline(foot_points_soc, foot_points_y)
        cs_z = CubicSpline(foot_points_soc, foot_points_z)

        # 生成更平滑的曲线
        soc_smooth = np.linspace(min(foot_points_soc), max(foot_points_soc), 100)
        y_smooth = cs_y(soc_smooth)
        z_smooth = cs_z(soc_smooth)
        x_smooth = [log_max] * len(soc_smooth)

        # 绘制连接曲线
        ax.plot(x_smooth, y_smooth, z_smooth,
                color='red',
                linestyle='-',
                linewidth=2.5,
                alpha=0.9,
                zorder=5)

    # 设置坐标轴标签
    ax.set_xlabel('Relaxation Time $\\tau$ (s)', fontsize=14, labelpad=15)
    ax.set_ylabel('SOC (%)', fontsize=14, labelpad=15)
    ax.set_zlabel('$\\gamma$ (m$\\Omega$·cm$^2$)', fontsize=14, labelpad=0)

    # 设置Y轴刻度和范围 - 使用SOC值
    soc_min, soc_max = min(soc_values), max(soc_values)
    soc_range = soc_max - soc_min
    ax.set_yticks(soc_values)
    ax.set_yticklabels([f'{int(soc)}%' for soc in soc_values], fontsize=11)
    ax.set_ylim(max(0, soc_min - soc_range * 0.1), soc_max + soc_range * 0.1)

    # 设置X轴刻度标签（对数刻度已在之前计算）
    xticks = np.arange(log_min, log_max + 1)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f'$10^{{{i}}}$' for i in xticks], fontsize=11)
    ax.set_xlim(log_min, log_max)

    # 设置视角 - 调整以增强3D效果
    ax.view_init(elev=27, azim=-135)


    # 添加网格 - 增加透明度
    # ax.grid(False)

    # 设置坐标轴范围 - 确保有明显的3D效果
    all_gamma = [np.max(analyzer.gamma[analyzer.gamma > 0]) for analyzer in analyzers]
    ax.set_zlim(0, max(all_gamma) * 1000 * 1.3)

    # 设置3D框的纵横比，拉大Y轴间距（Y轴是X轴的1.5倍）
    ax.set_box_aspect((1, 1.5, 0.75))

    # 设置空标题并移除标题空间
    ax.set_title('', pad=0)

    # 调整子图边距，减小上下留白
    plt.subplots_adjust(top=0.95, bottom=0.05, left=0.05, right=0.95)

    # 添加图例
    # ax.legend(loc='upper right', fontsize=11, bbox_to_anchor=(1.25, 1.0))

    # 添加XY平面底板 - 使用SOC范围
    x_plane = np.linspace(log_min, log_max, 20)
    y_plane = np.linspace(max(0, soc_min - soc_range * 0.1), soc_max + soc_range * 0.1, 2)

    X_plane, Y_plane = np.meshgrid(x_plane, y_plane)
    Z_plane = np.zeros_like(X_plane)

    ax.plot_surface(X_plane, Y_plane, Z_plane,
                    alpha=0.05,
                    color='blue')

    # 保存图像
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        fig.savefig(save_path, dpi=600, bbox_inches='tight', pad_inches=0.0)
        print(f"\n对比图已保存至: {save_path}")

    # 保存峰值数据到CSV
    if csv_save_path and peak_data:
        os.makedirs(os.path.dirname(csv_save_path) or '.', exist_ok=True)
        import csv
        with open(csv_save_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=['SOC', 'peak_gamma_mOhm_cm2', 'tau_peak_s', 'log_tau_peak'])
            writer.writeheader()
            for row in peak_data:
                writer.writerow(row)
        print(f"\n峰值数据已保存至: {csv_save_path}")

    # 收集并保存拟合误差数据
    error_data = []
    for soc, label, analyzer in zip(soc_values, labels, analyzers):
        rmse = analyzer.calculate_rmse() * 1000
        mape = analyzer.calculate_mape()
        r2_re, r2_im = analyzer.calculate_r_squared()
        error_data.append({
            'SOC': soc,
            'label': label,
            'RMSE_mOhm_cm2': rmse,
            'MAPE_percent': mape,
            'R2_Real': r2_re,
            'R2_Imaginary': r2_im,
            'lambda_opt': analyzer.lambda_opt
        })

    # 默认保存路径设置
    if error_csv_save_path is None:
        error_csv_save_path = r"E:\PycharmProjects\PythonProject1\output\DRT_fitting_errors.csv"

    # 保存误差数据到CSV
    if error_data:
        os.makedirs(os.path.dirname(error_csv_save_path) or '.', exist_ok=True)
        import csv
        with open(error_csv_save_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=['SOC', 'label', 'RMSE_mOhm_cm2', 'MAPE_percent', 'R2_Real', 'R2_Imaginary', 'lambda_opt'])
            writer.writeheader()
            for row in error_data:
                writer.writerow(row)
        print(f"\n拟合误差数据已保存至: {error_csv_save_path}")

    # 打印统计信息
    print("\n" + "="*70)
    print("Fitting Statistics Summary")
    print("="*70)
    for row in error_data:
        print(f"\nSOC {int(row['SOC'])}% ({row['label']}):")
        print(f"  RMSE: {row['RMSE_mOhm_cm2']:.4f} mOhm·cm^2")
        print(f"  MAPE: {row['MAPE_percent']:.2f}%")
        print(f"  R2 (Real): {row['R2_Real']:.4f}")
        print(f"  R2 (Imaginary): {row['R2_Imaginary']:.4f}")
        print(f"  Lambda: {row['lambda_opt']:.3e}")

    print("\n" + "="*70)

    return fig


if __name__ == "__main__":
    # ==================== 使用方式 ====================
    # 绘制3D对比图
    folder_path = (r"E:\Datasets\314Ah\Bat2\EIS4SOC\17degC")
    pattern = "*D*1h*.txt"
    label_pattern = r"1h-(\d+)A"
    save_path = r"E:\PycharmProjects\PythonProject1\output\DRT_3D_SOC-17.png"
    csv_save_path = r"E:\PycharmProjects\PythonProject1\output\DRT_peak_data-test17.csv"

    fig = plot_drt_batch_3d(folder_path, pattern=pattern,
                           label_pattern=label_pattern,
                           save_path=None,
                           csv_save_path=csv_save_path,
                           # error_csv_save_path 不指定时默认保存到output/DRT_fitting_errors.csv
                           error_csv_save_path=None,
                           view_elev=27,
                           view_azim=-135)

    plt.show()
