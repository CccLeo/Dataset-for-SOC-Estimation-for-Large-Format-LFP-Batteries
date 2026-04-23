"""DRT分析命令行工具

对单个EIS数据文件进行DRT分析并生成结果报告
"""
import sys
sys.path.append("E:/PycharmProjects/Paper1")
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# 导入核心库
from battery_analysis.drt import DRTAnalyzerFinal, read_and_preprocess_eis, plot_detailed_results, plot_2x2_results


if __name__ == "__main__":
    file_path = r"E:\Datasets\314Ah\Bat4\RPT1\EIS对比\#4-RPT1-D75%-16h.txt"

    print("="*70)
    print("Final Optimized DRT Analysis")
    print("="*70)

    # 读取数据
    freq, Z_re, Z_im, Rs = read_and_preprocess_eis(file_path)

    print("\n" + "="*70)
    print("Running Optimized DRT Fit")
    print("="*70)

    # 创建分析器并拟合
    analyzer = DRTAnalyzerFinal(freq, Z_re, Z_im, n_tau=200)
    analyzer.fit_optimized(lambda_reg=None)

    # 计算误差指标
    rmse = analyzer.calculate_rmse() * 1000
    mape = analyzer.calculate_mape()
    r2_re, r2_im = analyzer.calculate_r_squared()

    print("\n" + "="*70)
    print("Final Results")
    print("="*70)
    print(f"RMSE: {rmse:.4f} mOhm*cm^2")
    print(f"MAPE: {mape:.2f}%")
    print(f"R2 (Real): {r2_re:.4f}")
    print(f"R2 (Imag): {r2_im:.4f}")
    print(f"Lambda: {analyzer.lambda_opt:.3e}")

    # 分析DRT峰
    if analyzer.gamma is not None:
        peaks, properties = find_peaks(analyzer.gamma, height=np.max(analyzer.gamma)*0.05)
        if len(peaks) > 0:
            print(f"\nDetected {len(peaks)} DRT peaks:")
            print("-" * 70)
            sorted_peaks = peaks[np.argsort(-analyzer.gamma[peaks])]
            for i, peak in enumerate(sorted_peaks):
                tau_val = analyzer.tau[peak]
                gamma_val = analyzer.gamma[peak]

                if tau_val < 0.01:
                    process = "Fast kinetics"
                elif tau_val < 1:
                    process = "Charge transfer / SEI"
                elif tau_val < 10:
                    process = "Diffusion (medium)"
                else:
                    process = "Diffusion (slow)"

                print(f"Peak {i+1}: tau={tau_val:.3e}s, gamma={gamma_val*1000:.3f} mOhm*cm^2 -> {process}")

    # 绘图
    print("\nGenerating detailed plots...")
    fig, axes = plot_detailed_results(analyzer, title="Final Optimized DRT Analysis - " + file_path.split('\\')[-1])

    # 生成2×2学术风格图
    print("\nGenerating 2×2 academic style plots...")
    fig_2x2, axes_2x2 = plot_2x2_results(analyzer, title="DRT Analysis Results")

    # 保存结果
    output_dir = r"E:\PycharmProjects\PythonProject1\output"
    os.makedirs(output_dir, exist_ok=True)

    # 保存详细图像
    fig_file = os.path.join(output_dir, 'DRT_final_analysis.png')
    fig.savefig(fig_file, dpi=150, bbox_inches='tight')
    print(f"\nDetailed plot saved to: {fig_file}")

    # 保存2×2图像
    fig_2x2_file = os.path.join(output_dir, 'DRT_2x2_analysis.png')
    fig_2x2.savefig(fig_2x2_file, dpi=300, bbox_inches='tight')
    print(f"2×2 academic plot saved to: {fig_2x2_file}")

    # 保存DRT数据
    result_df = pd.DataFrame({
        'tau(s)': analyzer.tau,
        'gamma(Ohm*cm^2)': analyzer.gamma
    })
    output_file = os.path.join(output_dir, 'DRT_final_result.csv')
    result_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"DRT data saved to: {output_file}")

    # 保存拟合数据
    fit_df = pd.DataFrame({
        'freq(Hz)': analyzer.freq,
        'Z_re_exp(Ohm*cm^2)': analyzer.Z_re,
        'Z_im_exp(Ohm*cm^2)': analyzer.Z_im,
        'Z_re_fit(Ohm*cm^2)': analyzer.Z_fit_re,
        'Z_im_fit(Ohm*cm^2)': analyzer.Z_fit_im,
    })
    fit_file = os.path.join(output_dir, 'DRT_final_fit.csv')
    fit_df.to_csv(fit_file, index=False, encoding='utf-8-sig')
    print(f"Fit data saved to: {fit_file}")

    print("\n" + "="*70)
    print("Analysis Complete!")
    print("="*70)

    plt.show()
