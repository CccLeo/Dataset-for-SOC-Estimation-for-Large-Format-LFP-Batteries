"""DRT plotting module

Provides visualization functions for DRT analysis results, including detailed result plots and academic-style 2×2 plots
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from matplotlib.ticker import FuncFormatter

# Font and minus sign display settings
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False  # Use ASCII minus sign instead of Unicode minus sign
plt.rcParams['mathtext.default'] = 'regular'  # Disable LaTeX-style mathematical symbol rendering


def plot_detailed_results(analyzer, title="DRT Analysis Results"):
    """Plot detailed results"""

    # Custom tick formatter function, replace Unicode minus sign with regular minus sign
    def fix_minus_formatter(x, pos):
        """Format tick labels, replace Unicode minus sign with regular minus sign"""
        if x == 0:
            return '0'
        if x >= 1:
            return f'{x:.0f}'
        # For numbers less than 1, display as power of 10
        exponent = int(np.floor(np.log10(abs(x))))
        base = x / (10 ** exponent)
        return f'{base:.0g}×10^{{{exponent}}}'

    def fix_minus_tex_formatter(x, pos):
        """Use LaTeX format but avoid Unicode minus sign"""
        if x == 0:
            return r'$0$'
        if x >= 1:
            return f'${x:.0f}$'
        exponent = int(np.floor(np.log10(abs(x))))
        return f'$10^{{{exponent}}}$'

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # 1. Nyquist plot
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(analyzer.Z_re * 1000, analyzer.Z_im * 1000, 'o-',
            label='Experimental', markersize=6, linewidth=2, color='blue')
    if analyzer.Z_fit_re is not None:
        ax1.plot(analyzer.Z_fit_re * 1000, analyzer.Z_fit_im * 1000, '--',
                label='DRT Fit', linewidth=2, color='red', alpha=0.8)

    ax1.set_xlabel("Z' (mOhm*cm^2)", fontsize=11)
    ax1.set_ylabel("-Z'' (mOhm*cm^2)", fontsize=11)
    ax1.set_title('Nyquist Plot', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # 2. DRT spectrum plot
    ax2 = fig.add_subplot(gs[0, 1])
    if analyzer.gamma is not None:
        ax2.semilogx(analyzer.tau, analyzer.gamma * 1000, 'b-', linewidth=2.5)
        ax2.fill_between(analyzer.tau, analyzer.gamma * 1000, alpha=0.3)
        ax2.set_xlabel("Relaxation Time tau (s)", fontsize=11)
        ax2.set_ylabel("Gamma (mOhm*cm^2)", fontsize=11)
        ax2.set_title('Distribution of Relaxation Times', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        # Fix x-axis ticks
        ax2.xaxis.set_major_formatter(FuncFormatter(fix_minus_tex_formatter))

        # Mark peaks
        peaks, _ = find_peaks(analyzer.gamma, height=np.max(analyzer.gamma)*0.05)
        if len(peaks) > 0:
            sorted_peaks = peaks[np.argsort(-analyzer.gamma[peaks])][:5]
            for peak in sorted_peaks:
                ax2.annotate(f'{analyzer.tau[peak]:.2g}s',
                           xy=(analyzer.tau[peak], analyzer.gamma[peak] * 1000),
                           xytext=(10, 10), textcoords='offset points',
                           fontsize=8, bbox=dict(boxstyle='round,pad=0.3',
                                                 facecolor='yellow', alpha=0.7),
                           arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    # 3. Real part fitting comparison
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.semilogx(analyzer.freq, analyzer.Z_re * 1000, 'o-',
                label='Experimental', markersize=6, linewidth=2, color='blue')
    if analyzer.Z_fit_re is not None:
        ax3.semilogx(analyzer.freq, analyzer.Z_fit_re * 1000, '--',
                    label='DRT Fit', linewidth=2, color='red', alpha=0.8)

    ax3.set_xlabel("Frequency (Hz)", fontsize=11)
    ax3.set_ylabel("-Z' (mOhm*cm^2)", fontsize=11)
    ax3.set_title('Real Part Comparison', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    # Fix x-axis ticks
    ax3.xaxis.set_major_formatter(FuncFormatter(fix_minus_tex_formatter))

    # 4. Imaginary part fitting comparison
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.semilogx(analyzer.freq, analyzer.Z_im * 1000, 'o-',
                label='Experimental', markersize=6, linewidth=2, color='blue')
    if analyzer.Z_fit_im is not None:
        ax4.semilogx(analyzer.freq, analyzer.Z_fit_im * 1000, '--',
                    label='DRT Fit', linewidth=2, color='red', alpha=0.8)

    ax4.set_xlabel("Frequency (Hz)", fontsize=11)
    ax4.set_ylabel("-Z'' (mOhm*cm^2)", fontsize=11)
    ax4.set_title('Imaginary Part Comparison', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    # Fix x-axis ticks
    ax4.xaxis.set_major_formatter(FuncFormatter(fix_minus_tex_formatter))

    # 5. Relative error distribution
    ax5 = fig.add_subplot(gs[1, 1])
    if analyzer.Z_fit_re is not None:
        residual_re = (analyzer.Z_fit_re - analyzer.Z_re) / (np.abs(analyzer.Z_re) + 1e-10) * 100
        residual_im = (analyzer.Z_fit_im - analyzer.Z_im) / (np.abs(analyzer.Z_im) + 1e-10) * 100

        ax5.semilogx(analyzer.freq, residual_re, 'o-', label='Real', markersize=5, linewidth=1.5)
        ax5.semilogx(analyzer.freq, residual_im, 's-', label='Imag', markersize=5, linewidth=1.5)
        ax5.axhline(y=0, color='k', linestyle='--', linewidth=0.5)

    ax5.set_xlabel("Frequency (Hz)", fontsize=11)
    ax5.set_ylabel("Relative Error (%)", fontsize=11)
    ax5.set_title('Relative Error Distribution', fontsize=12, fontweight='bold')
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3)
    # Fix x-axis ticks
    ax5.xaxis.set_major_formatter(FuncFormatter(fix_minus_tex_formatter))

    # 6. Fitting statistics
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')

    if analyzer.Z_fit_re is not None:
        rmse = analyzer.calculate_rmse() * 1000
        mape = analyzer.calculate_mape()
        r2_re, r2_im = analyzer.calculate_r_squared()

        stats_text = f"""
        Fitting Statistics

        RMSE: {rmse:.4f} mOhm*cm^2
        MAPE: {mape:.2f}%

        R² (Real): {r2_re:.4f}
        R² (Imag): {r2_im:.4f}

        Lambda: {analyzer.lambda_opt:.3e}
        Data Points: {len(analyzer.freq)}
        Tau Points: {analyzer.n_tau}

        Tau Range: {analyzer.tau.min():.3e} - {analyzer.tau.max():.3e} s
        """

        ax6.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
                verticalalignment='center', bbox=dict(boxstyle='round',
                                                      facecolor='wheat', alpha=0.5))

    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)

    return fig, [ax1, ax2, ax3, ax4, ax5, ax6]


def plot_2x2_results(analyzer, title="DRT Analysis Results"):
    """
    2×2 layout plot based on plot_detailed_results style, removed relative error and statistics panels
    Subplot order: first row [Nyquist plot, DRT spectrum plot], second row [real part fitting comparison, imaginary part fitting comparison]
    Fully retains the plotting style and parameters of the original plot_detailed_results
    """
    # Custom tick formatter function, exactly the same as plot_detailed_results
    def fix_minus_tex_formatter(x, pos):
        """Use LaTeX format but avoid Unicode minus sign, same as original function"""
        if x == 0:
            return r'$0$'
        if x >= 1:
            return f'${x:.0f}$'
        exponent = int(np.floor(np.log10(abs(x))))
        return f'$10^{{{exponent}}}$'

    # Create 2×2 layout, similar size to original plot_detailed_results, increase vertical spacing to fit titles below
    fig = plt.figure(figsize=(10, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.2, wspace=0.2)

    # -------------------------- First Row --------------------------
    # 1. Nyquist plot (exactly the same as plot_detailed_results #1)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(analyzer.Z_re * 1000, analyzer.Z_im * 1000, 'o-',
            label='Experimental', markersize=6, linewidth=2, color='blue')
    if analyzer.Z_fit_re is not None:
        ax1.plot(analyzer.Z_fit_re * 1000, analyzer.Z_fit_im * 1000, '--',
                label='DRT Fit', linewidth=2, color='red', alpha=0.8)

    ax1.set_xlabel("Z' (mΩ·cm²)", fontsize=11)
    ax1.set_ylabel("-Z'' (mΩ·cm²)", fontsize=11)
    ax1.set_title('(a) Nyquist Plot', fontsize=12, y=-0.18)
    ax1.legend(fontsize=14)
    ax1.grid(True, alpha=0.3)

    # 2. DRT spectrum plot (exactly the same as plot_detailed_results #2)
    ax2 = fig.add_subplot(gs[0, 1])
    if analyzer.gamma is not None:
        ax2.semilogx(analyzer.tau, analyzer.gamma * 1000, 'b-', linewidth=2.5)
        ax2.fill_between(analyzer.tau, analyzer.gamma * 1000, alpha=0.3)
        ax2.set_xlabel("Relaxation Time $\\tau$ (s)", fontsize=11)
        ax2.set_ylabel("γ (mΩ·cm²)", fontsize=11)
        ax2.set_title('(b) Distribution of Relaxation Times', fontsize=12, y=-0.18)
        ax2.grid(True, alpha=0.3)
        # Fix x-axis ticks
        ax2.xaxis.set_major_formatter(FuncFormatter(fix_minus_tex_formatter))

        # Mark peaks (exactly the same as original function, show up to 5 peaks)
        peaks, _ = find_peaks(analyzer.gamma, height=np.max(analyzer.gamma)*0.05)
        if len(peaks) > 0:
            sorted_peaks = peaks[np.argsort(-analyzer.gamma[peaks])][:5]
            for peak in sorted_peaks:
                ax2.annotate(f'{analyzer.tau[peak]:.2g}s',
                           xy=(analyzer.tau[peak], analyzer.gamma[peak] * 1000),
                           xytext=(10, 10), textcoords='offset points',
                           fontsize=8, bbox=dict(boxstyle='round,pad=0.3',
                                                 facecolor='yellow', alpha=0.7),
                           arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    # -------------------------- Second Row --------------------------
    # 3. Real part fitting comparison (exactly the same as plot_detailed_results #3)
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.semilogx(analyzer.freq, analyzer.Z_re * 1000, 'o-',
                label='Experimental', markersize=6, linewidth=2, color='blue')
    if analyzer.Z_fit_re is not None:
        ax3.semilogx(analyzer.freq, analyzer.Z_fit_re * 1000, '--',
                    label='DRT Fit', linewidth=2, color='red', alpha=0.8)

    ax3.set_xlabel("Frequency (Hz)", fontsize=11)
    ax3.set_ylabel("-Z' (mΩ·cm²)", fontsize=11)
    ax3.set_title('(c) Real Part Comparison', fontsize=12, y=-0.18)
    ax3.legend(fontsize=14)
    ax3.grid(True, alpha=0.3)
    # Fix x-axis ticks
    ax3.xaxis.set_major_formatter(FuncFormatter(fix_minus_tex_formatter))

    # 4. Imaginary part fitting comparison (exactly the same as plot_detailed_results #4)
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.semilogx(analyzer.freq, analyzer.Z_im * 1000, 'o-',
                label='Experimental', markersize=6, linewidth=2, color='blue')
    if analyzer.Z_fit_im is not None:
        ax4.semilogx(analyzer.freq, analyzer.Z_fit_im * 1000, '--',
                    label='DRT Fit', linewidth=2, color='red', alpha=0.8)

    ax4.set_xlabel("Frequency (Hz)", fontsize=11)
    ax4.set_ylabel("-Z'' (mΩ·cm²)", fontsize=11)
    ax4.set_title('(d) Imaginary Part Comparison', fontsize=12, y=-0.18)
    ax4.legend(fontsize=14)
    ax4.grid(True, alpha=0.3)
    # Fix x-axis ticks
    ax4.xaxis.set_major_formatter(FuncFormatter(fix_minus_tex_formatter))

    # Main title same as original function
    # fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)

    # Adjust overall margins, reduce top and bottom whitespace
    plt.subplots_adjust(top=0.95, bottom=0.08, left=0.1, right=0.95)

    return fig, [ax1, ax2, ax3, ax4]
