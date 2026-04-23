"""EIS plotting module"""

import numpy as np
import matplotlib.pyplot as plt
import os

# Low frequency threshold (Hz)
LOW_FREQ_THRESHOLD = 1.0


def plot_results(freq, Z_re, Z_im, Z_fit, metrics, title, save_path=None):
    """Plot fitting results (multiple subplots)

    Args:
        freq: frequency array
        Z_re: real part array
        Z_im: imaginary part array
        Z_fit: fitted impedance array
        metrics: fitting metrics dictionary
        title: plot title
        save_path: save path (optional)

    Returns:
        matplotlib figure object
    """
    fig = plt.figure(figsize=(16, 10))

    lowfreq_mask = freq < LOW_FREQ_THRESHOLD

    # 1. Nyquist plot
    ax1 = plt.subplot(2, 4, 1)
    ax1.plot(Z_re, -Z_im, 'o', label='Exp', markersize=8, alpha=0.7, color='blue',
             markeredgewidth=1.5, markeredgecolor='darkblue')
    ax1.plot(Z_fit.real, -Z_fit.imag, '-', label='Fit', linewidth=3, color='red')
    ax1.scatter(Z_re[lowfreq_mask], -Z_im[lowfreq_mask], s=100, c='gold',
               edgecolors='black', linewidths=2, zorder=10, label='Low freq')
    ax1.set_xlabel("Z' (Ohm*cm^2)", fontsize=11, fontweight='bold')
    ax1.set_ylabel("-Z'' (Ohm*cm^2)", fontsize=11, fontweight='bold')
    ax1.set_title('Nyquist Plot', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')

    # 2. Real part comparison
    ax2 = plt.subplot(2, 4, 2)
    ax2.semilogx(freq, Z_re, 'o', label='Exp', markersize=5, color='blue')
    ax2.semilogx(freq, Z_fit.real, '-', label='Fit', linewidth=2, color='red')
    ax2.axvspan(0, 1, alpha=0.2, color='yellow')
    ax2.set_xlabel('Freq (Hz)', fontsize=10)
    ax2.set_ylabel("Z' (Ohm*cm^2)", fontsize=10)
    ax2.set_title('Real Part', fontsize=11, fontweight='bold')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # 3. Imaginary part comparison
    ax3 = plt.subplot(2, 4, 3)
    ax3.semilogx(freq, -Z_im, 'o', label='Exp', markersize=5, color='blue')
    ax3.semilogx(freq, -Z_fit.imag, '-', label='Fit', linewidth=2, color='red')
    ax3.axvspan(0, 1, alpha=0.2, color='yellow')
    ax3.set_xlabel('Freq (Hz)', fontsize=10)
    ax3.set_ylabel("-Z'' (Ohm*cm^2)", fontsize=10)
    ax3.set_title('Imaginary Part', fontsize=11, fontweight='bold')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # 4. Bode magnitude
    ax4 = plt.subplot(2, 4, 4)
    ax4.semilogx(freq, np.abs(Z_re + 1j*Z_im), 'o', label='Exp', markersize=5, color='blue')
    ax4.semilogx(freq, np.abs(Z_fit), '-', label='Fit', linewidth=2, color='red')
    ax4.axvspan(0, 1, alpha=0.2, color='yellow')
    ax4.set_xlabel('Freq (Hz)', fontsize=10)
    ax4.set_ylabel('|Z| (Ohm*cm^2)', fontsize=10)
    ax4.set_title('Bode - Magnitude', fontsize=11, fontweight='bold')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    # 5. Bode phase
    ax5 = plt.subplot(2, 4, 5)
    phase_exp = np.arctan2(Z_im, Z_re) * 180 / np.pi
    phase_fit = np.arctan2(Z_fit.imag, Z_fit.real) * 180 / np.pi
    ax5.semilogx(freq, phase_exp, 'o', label='Exp', markersize=5, color='blue')
    ax5.semilogx(freq, phase_fit, '-', label='Fit', linewidth=2, color='red')
    ax5.axvspan(0, 1, alpha=0.2, color='yellow')
    ax5.set_xlabel('Freq (Hz)', fontsize=10)
    ax5.set_ylabel('Phase (deg)', fontsize=10)
    ax5.set_title('Bode - Phase', fontsize=11, fontweight='bold')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)

    # 6. Real part error
    ax6 = plt.subplot(2, 4, 6)
    err_re = np.abs(Z_fit.real - Z_re) / (np.abs(Z_re) + 1e-10) * 100
    ax6.semilogx(freq, err_re, 'o-', linewidth=2, markersize=4, color='green')
    ax6.axvspan(0, 1, alpha=0.2, color='yellow')
    ax6.set_xlabel('Freq (Hz)', fontsize=10)
    ax6.set_ylabel('Error (%)', fontsize=10)
    ax6.set_title('Real Part Error', fontsize=11, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    ax6.set_yscale('log')

    # 7. Imaginary part error
    ax7 = plt.subplot(2, 4, 7)
    err_im = np.abs(Z_fit.imag - Z_im) / (np.abs(Z_im) + 1e-10) * 100
    ax7.semilogx(freq, err_im, 'o-', linewidth=2, markersize=4, color='purple')
    ax7.axvspan(0, 1, alpha=0.2, color='yellow')
    ax7.set_xlabel('Freq (Hz)', fontsize=10)
    ax7.set_ylabel('Error (%)', fontsize=10)
    ax7.set_title('Imag Part Error', fontsize=11, fontweight='bold')
    ax7.grid(True, alpha=0.3)
    ax7.set_yscale('log')

    # 8. Statistical information
    ax8 = plt.subplot(2, 4, 8)
    ax8.axis('off')
    stats = f"""
    STATISTICS

    Overall:
    MAPE Real: {metrics['mape_re']:.2f}%
    MAPE Imag: {metrics['mape_im']:.2f}%
    R2 Real: {metrics['r2_re']:.6f}
    R2 Imag: {metrics['r2_im']:.6f}

    Low Freq (<1Hz):
    MAPE Real: {metrics['mape_re_lf']:.2f}%
    MAPE Imag: {metrics['mape_im_lf']:.2f}%
    """
    ax8.text(0.1, 0.5, stats, fontsize=11, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.suptitle(f'{title}\nModel 2: Dual RC + Bounded Warburg',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nFigure saved: {save_path}")

    return fig


def plot_comparison_nyquist(results_with_data, output_dir, folder_name="EIS Batch"):
    """Plot Nyquist comparison for all EIS files

    Args:
        results_with_data: list containing (result_dict, (freq, Z_re, Z_im, Z_fit))
        output_dir: output directory
        folder_name: folder name (for title)

    Returns:
        save path
    """
    # Filter successful results
    successful_results = [(r, d) for r, d in results_with_data if r['status'] == 'success' and d is not None]

    if len(successful_results) == 0:
        print("  Warning: No successfully fitted data, cannot generate comparison plot")
        return None

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # Define color cycle (using unified plotting style colors)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
              '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5']

    # Plot each curve
    for i, (result_dict, (freq, Z_re, Z_im, Z_fit)) in enumerate(successful_results):
        color = colors[i % len(colors)]
        edge_color = colors[i % len(colors)]  # Use same color for edge

        # Generate legend label
        if 'battery_label' in result_dict and 'test_label' in result_dict:
            label = f"{result_dict['battery_label']}-{result_dict['test_label']}"
            if 'soc_pct' in result_dict:
                label += f"-{result_dict['soc_pct']:.0f}%"
        else:
            # Use filename (without extension)
            label = os.path.splitext(result_dict['filename'])[0]
            # Limit length
            if len(label) > 20:
                label = label[:17] + "..."

        # Low frequency marker (freq < 1Hz)
        lowfreq_mask = freq < LOW_FREQ_THRESHOLD

        # Plot experimental data (scatter) - using unified plotting style
        ax.plot(Z_re, -Z_im, 'o', label=label, markersize=10, alpha=0.7,
                color=color, markeredgewidth=1.5, markeredgecolor=edge_color, zorder=2)

        # Plot fitting curve (line) - using unified plotting style
        ax.plot(Z_fit.real, -Z_fit.imag, '-', linewidth=4, color=color, alpha=0.9, zorder=3)

        # Plot low frequency data point markers - use corresponding sample color fill
        ax.scatter(Z_re[lowfreq_mask], -Z_im[lowfreq_mask], s=100, c=color,
                   edgecolors='black', linewidths=2, zorder=10)

    # Set axis labels and title - using unified plotting style
    ax.set_xlabel("Z' (Ohm*cm^2)", fontsize=11, fontweight='bold')
    ax.set_ylabel("-Z'' (Ohm*cm^2)", fontsize=11, fontweight='bold')
    ax.set_title(f'Nyquist Plot Comparison - {folder_name}\nModel 2: Dual RC + Bounded Warburg',
                 fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')

    # Set legend - using unified plotting style
    ax.legend(fontsize=9, loc='best')

    plt.tight_layout()

    # Save figure
    save_path = os.path.join(output_dir, "EIS_Model2_Comparison_Nyquist.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nComparison plot saved: {save_path}")
    plt.close(fig)

    return save_path
