"""
EIS和DRT数据集成脚本
整合EIS_plot.py的实部/虚部提取功能和DRT-3D.py的peak_gamma提取功能
输出CSV文件包含：SOC, 实部大小, 虚部大小, peak_gamma大小
"""
import sys
sys.path.append("E:/PycharmProjects/PythonProject1")
import numpy as np
import pandas as pd
import os
import glob
import re
from scipy.signal import find_peaks
from battery_analysis.drt import DRTAnalyzerFinal, read_and_preprocess_eis
import warnings
warnings.filterwarnings('ignore')


def integrate_eis_drt_data(folder_path, pattern='*D*1h*.txt',
                          target_freq_re=0.01, target_freq_im=0.1,
                          output_csv_path=None):
    """
    集成EIS和DRT数据，输出包含SOC、实部、虚部和peak_gamma的CSV文件

    Args:
        folder_path: EIS数据文件夹路径
        pattern: 文件匹配模式，默认'*D*1h*.txt'
        target_freq_re: 实部目标频率（Hz），默认0.01
        target_freq_im: 虚部目标频率（Hz），默认0.1
        output_csv_path: 输出CSV文件路径，如果为None则默认保存到output文件夹

    Returns:
        bool: 是否成功
    """
    # 查找匹配的文件
    search_pattern = os.path.join(folder_path, pattern)
    files = sorted(glob.glob(search_pattern))

    if not files:
        print(f"未找到匹配的文件: {search_pattern}")
        return False

    print(f"找到 {len(files)} 个文件:")
    for f in files:
        print(f"  - {os.path.basename(f)}")

    # 存储所有数据
    all_data = []

    for file_path in files:
        basename = os.path.basename(file_path)
        print(f"\n处理文件: {basename}")

        try:
            # ==================== 1. 提取SOC值 ====================
            soc_match = re.search(r"D(\d+)", basename)
            if soc_match:
                soc = float(soc_match.group(1))
                print(f"  SOC: {soc}%")
            else:
                print(f"  警告：无法从文件名提取SOC值，跳过该文件")
                continue

            # ==================== 2. 提取特定频率下的实部和虚部（原始数据） ====================
            # 直接读取原始CSV数据，不进行校正
            df = pd.read_csv(file_path, skiprows=1, delimiter=r'\s+')
            freq = df['Freq(Hz)'].to_numpy()
            Z_re_original = df["Z'(Ohm.cm²)"].to_numpy()
            Z_im_original = df["Z''(Ohm.cm²)"].to_numpy()

            # 找到最接近实部目标频率的数据点
            idx_re = (np.abs(freq - target_freq_re)).argmin()
            actual_freq_re = freq[idx_re]
            Z_re_value = Z_re_original[idx_re]

            # 检查实部频率误差（允许50%误差）
            if abs(actual_freq_re - target_freq_re) / target_freq_re > 0.5:
                print(f"  警告：未找到接近 {target_freq_re} Hz 的实部频率点，实际: {actual_freq_re:.4f} Hz")

            # 找到最接近虚部目标频率的数据点
            idx_im = (np.abs(freq - target_freq_im)).argmin()
            actual_freq_im = freq[idx_im]
            Z_im_value = Z_im_original[idx_im]

            # 检查虚部频率误差（允许50%误差）
            if abs(actual_freq_im - target_freq_im) / target_freq_im > 0.5:
                print(f"  警告：未找到接近 {target_freq_im} Hz 的虚部频率点，实际: {actual_freq_im:.4f} Hz")

            print(f"  实部 @ {actual_freq_re:.3f} Hz: {Z_re_value:.6f} Ohm.cm2")
            print(f"  虚部 @ {actual_freq_im:.3f} Hz: {Z_im_value:.6f} Ohm.cm2")

            # ==================== 3. DRT分析提取peak_gamma ====================
            # 使用read_and_preprocess_eis读取并预处理数据（会自动减去Rs）
            freq_drt, Z_re_drt, Z_im_drt, Rs = read_and_preprocess_eis(file_path)

            # 创建DRT分析器并拟合
            analyzer = DRTAnalyzerFinal(freq_drt, Z_re_drt, Z_im_drt, n_tau=200)
            analyzer.fit_optimized(lambda_reg=0.3e4)  # 使用固定lambda，和DRT-3D.py保持一致

            # 提取gamma数据
            gamma = analyzer.gamma
            tau = analyzer.tau

            # 只考虑正的gamma值
            mask = gamma > 0
            gamma_valid = gamma[mask]
            tau_valid = tau[mask]

            if len(gamma_valid) == 0:
                print(f"  警告：未找到有效的gamma值，跳过该文件")
                continue

            # 找到峰值
            peaks, _ = find_peaks(gamma_valid, height=np.max(gamma_valid)*0.05)
            if len(peaks) == 0:
                print(f"  警告：未检测到DRT峰值，跳过该文件")
                continue

            # 找到最高峰值
            max_peak_idx = peaks[np.argmax(gamma_valid[peaks])]
            peak_gamma_value = gamma_valid[max_peak_idx] * 1000  # 转换为mOhm·cm²
            tau_peak = tau_valid[max_peak_idx]

            print(f"  峰值gamma: {peak_gamma_value:.6f} mOhm.cm2 @ tau={tau_peak:.3e} s")

            # ==================== 4. 保存数据 ====================
            all_data.append({
                'SOC': soc,
                'Z_re_Ohm_cm2': Z_re_value,
                'Z_im_Ohm_cm2': Z_im_value,
                'peak_gamma_mOhm_cm2': peak_gamma_value,
                'actual_freq_re_Hz': actual_freq_re,
                'actual_freq_im_Hz': actual_freq_im,
                'tau_peak_s': tau_peak
            })

        except Exception as e:
            print(f"  处理失败: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    if not all_data:
        print("\n没有有效的数据可以保存")
        return False

    # 按SOC排序
    all_data_sorted = sorted(all_data, key=lambda x: x['SOC'])

    # 设置默认输出路径
    if output_csv_path is None:
        output_dir = r"E:\PycharmProjects\PythonProject1\output"
        os.makedirs(output_dir, exist_ok=True)
        output_csv_path = os.path.join(output_dir, 'integrated_EIS_DRT_data.csv')

    # 保存到CSV文件
    # 输出格式：第一列SOC，第二列实部，第三列虚部，第四列peak_gamma
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as f:
        # 写入表头（可选，用户需要的是数据列）
        # f.write("SOC,Z_re_Ohm_cm2,Z_im_Ohm_cm2,peak_gamma_mOhm_cm2\n")
        for data in all_data_sorted:
            f.write(f"{data['SOC']:.1f},{data['Z_re_Ohm_cm2']:.10f},{data['Z_im_Ohm_cm2']:.10f},{data['peak_gamma_mOhm_cm2']:.10f}\n")

    print(f"\n集成数据已保存至: {output_csv_path}")
    print(f"数据格式：SOC(%), 实部(Ohm.cm2), 虚部(Ohm.cm2), peak_gamma(mOhm.cm2)")

    # 打印数据预览
    print("\n数据预览：")
    print("-" * 80)
    print(f"{'SOC':<6} {'Z_re':<18} {'Z_im':<18} {'peak_gamma':<18}")
    print("-" * 80)
    for data in all_data_sorted:
        print(f"{data['SOC']:<6.1f} {data['Z_re_Ohm_cm2']:<18.6f} {data['Z_im_Ohm_cm2']:<18.6f} {data['peak_gamma_mOhm_cm2']:<18.6f}")
    print("-" * 80)

    return True


if __name__ == "__main__":
    # 配置参数
    folder_path = r"E:\Datasets\314Ah\Bat2\EIS4SOC\17degC"  # EIS数据文件夹路径
    pattern = "*D*1h*.txt"  # 文件匹配模式
    target_freq_re = 0.01  # 实部目标频率（Hz）
    target_freq_im = 0.1  # 虚部目标频率（Hz）
    output_csv_path = r"E:\PycharmProjects\PythonProject1\output\17degC_integrated_data.csv"  # 输出文件路径

    # 运行集成
    success = integrate_eis_drt_data(
        folder_path=folder_path,
        pattern=pattern,
        target_freq_re=target_freq_re,
        target_freq_im=target_freq_im,
        output_csv_path=output_csv_path
    )

    if success:
        print("\n集成完成！")
    else:
        print("\n集成失败！")
