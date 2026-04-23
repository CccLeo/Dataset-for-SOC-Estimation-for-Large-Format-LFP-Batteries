import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import pandas as pd
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Global configuration constants
LOW_FREQ_THRESHOLD = 1.0  # Low frequency threshold (Hz)
DEFAULT_OUTPUT_DIR = r"E:\PycharmProjects\PythonProject1\output"

def read_eis_data(file_path, skiprows=1, filter_negative_im=True):
    """Read EIS data"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    data = pd.read_csv(file_path, skiprows=skiprows, delimiter=r'\s+')
    freq = data['Freq(Hz)'].to_numpy()
    Z_re = data["Z'(Ohm.cm²)"].to_numpy()
    Z_im = data["Z''(Ohm.cm²)"].to_numpy()

    if filter_negative_im:
        mask = Z_im < 0
        freq = freq[mask]
        Z_re = Z_re[mask]
        Z_im = Z_im[mask]

    return freq, Z_re, Z_im

def smart_initial_guess(freq, Z_re, Z_im):
    """Smart initial parameter estimation"""
    idx_min_im = np.argmin(np.abs(Z_im))
    Rs = Z_re[idx_min_im]
    R_total = np.max(Z_re) - np.min(Z_re)

    min_im_idx = np.argmin(Z_im)
    f_char = freq[min_im_idx]
    omega_char = 2 * np.pi * f_char

    Q = 1 / (R_total * (omega_char ** 0.85))
    alpha = 0.88
    sigma = 1e-4

    return {'Rs': Rs, 'R_total': R_total, 'Q': Q, 'alpha': alpha, 'sigma': sigma}

def get_initial_params(init_dict):
    """Generate initial parameters for Model 2 based on initial estimates"""
    return [
        init_dict['Rs']*0.9,
        init_dict['R_total']*0.35,
        init_dict['Q']*2,
        0.92,
        init_dict['R_total']*0.55,
        init_dict['Q']*0.6,
        0.88,
        init_dict['sigma']*3,
        0.5,
        0.1
    ]

def circuit_model2(params, omega):
    """Model 2: Rs + (R1||CPE1) + (R2||CPE2) + Bounded Warburg"""
    Rs, R1, Q1, a1, R2, Q2, a2, sigma, A, B = params
    Z1 = 1 / (1/R1 + (Q1 * (1j*omega)**a1))
    Z2 = 1 / (1/R2 + (Q2 * (1j*omega)**a2))
    s = np.sqrt(1j * omega)
    Zw = sigma * np.tanh(A * s) / (s * (1 + B * s))
    return Rs + Z1 + Z2 + Zw

def calculate_lowfreq_weights(freq, Z_re, Z_im, lowfreq_threshold=0.5):
    """Calculate weights emphasizing low frequencies"""
    weights_re = 1.0 / (np.abs(Z_re) + 1e-10)
    weights_im = 1.0 / (np.abs(Z_im) + 1e-10)

    lowfreq_mask = freq < lowfreq_threshold
    enhancement = np.ones_like(freq)
    enhancement[lowfreq_mask] = 5.0

    freq_weight = 1.0 + (np.max(freq) - freq) / (np.max(freq) - np.min(freq) + 1e-10)
    freq_weight = freq_weight ** 2

    weights_re = weights_re * enhancement * freq_weight
    weights_im = weights_im * enhancement * freq_weight

    weights_re = weights_re / np.mean(weights_re)
    weights_im = weights_im / np.mean(weights_im)

    return np.concatenate([weights_re, weights_im])

def fit_model_optimized(params_init, freq, Z_re, Z_im):
    """Fit Model 2"""
    omega = 2 * np.pi * freq
    weights = calculate_lowfreq_weights(freq, Z_re, Z_im, lowfreq_threshold=3.5)

    def objective_function(p):
        try:
            Z = circuit_model2(p, omega)
            res = np.concatenate([Z.real - Z_re, Z.imag - Z_im])
            return res * weights
        except:
            return np.ones(2 * len(freq)) * 1e6

    n_params = len(params_init)
    lower_bounds = np.ones(n_params) * 1e-12
    upper_bounds = np.ones(n_params) * 1e6

    for i in range(n_params):
        if 0.5 <= params_init[i] <= 1.0 and i > 2:
            lower_bounds[i] = 0.5
            upper_bounds[i] = 1.0

    result = least_squares(
        objective_function,
        params_init,
        bounds=(lower_bounds, upper_bounds),
        method='trf',
        loss='soft_l1',
        max_nfev=50000,
        verbose=0
    )

    return result

def calculate_metrics(Z_fit, Z_re, Z_im, freq):
    """Calculate fitting metrics"""
    mape_re = np.mean(np.abs(Z_fit.real - Z_re) / (np.abs(Z_re) + 1e-10)) * 100
    mape_im = np.mean(np.abs(Z_fit.imag - Z_im) / (np.abs(Z_im) + 1e-10)) * 100

    lowfreq_mask = freq < LOW_FREQ_THRESHOLD
    if np.sum(lowfreq_mask) > 0:
        mape_re_lf = np.mean(np.abs(Z_fit.real[lowfreq_mask] - Z_re[lowfreq_mask]) /
                            (np.abs(Z_re[lowfreq_mask]) + 1e-10)) * 100
        mape_im_lf = np.mean(np.abs(Z_fit.imag[lowfreq_mask] - Z_im[lowfreq_mask]) /
                            (np.abs(Z_im[lowfreq_mask]) + 1e-10)) * 100
    else:
        mape_re_lf, mape_im_lf = np.nan, np.nan

    r2_re = 1 - np.sum((Z_re - Z_fit.real)**2) / np.sum((Z_re - np.mean(Z_re))**2)
    r2_im = 1 - np.sum((Z_im - Z_fit.imag)**2) / np.sum((Z_im - np.mean(Z_im))**2)

    return {
        'mape_re': mape_re, 'mape_im': mape_im,
        'mape_re_lf': mape_re_lf, 'mape_im_lf': mape_im_lf,
        'r2_re': r2_re, 'r2_im': r2_im
    }

def plot_results(freq, Z_re, Z_im, Z_fit, metrics, title, save_path=None):
    """Plot fitting results"""
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

def parse_filename_metadata(filename):
    """Parse metadata from filename
    Naming convention: #4-RPT1-75%-21h.txt
    Returns: battery ID, test cycle, SOC, rest time
    """
    import re
    basename = os.path.basename(filename)
    name_without_ext = os.path.splitext(basename)[0]

    # 尝试解析
    pattern = r'#(\d+)-RPT(\d+)-(\d+)%-(\d+)h'
    match = re.search(pattern, name_without_ext)

    if match:
        return {
            'battery_id': int(match.group(1)),
            'test_cycle': int(match.group(2)),
            'soc_pct': float(match.group(3)),
            'rest_hours': float(match.group(4)),
            'battery_label': f"B{match.group(1)}",
            'test_label': f"RPT{match.group(2)}"
        }
    return {}

def save_parameters_to_csv(params, param_names, file_path, metadata={}):
    """Save parameters to CSV file"""
    import csv
    from datetime import datetime

    # Prepare data
    data = {'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    data.update(metadata)
    for name, val in zip(param_names, params):
        data[name] = val

    # Check if file exists
    file_exists = os.path.exists(file_path)

    # Write to CSV
    with open(file_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(data)

    print(f"Parameters saved to: {file_path}")

def fit_single_file(file_path, output_dir, plot_individual=False, return_data=False):
    """Fit single file

    Args:
        file_path: EIS file path
        output_dir: output directory
        plot_individual: whether to plot individual figures
        return_data: whether to return data needed for plotting (for comparison plots)

    Returns:
        result_dict: fitting result dictionary
        If return_data=True, additionally returns (freq, Z_re, Z_im, Z_fit)
    """
    try:
        freq, Z_re, Z_im = read_eis_data(file_path, filter_negative_im=True)
        filename = os.path.basename(file_path)

        # Parse metadata
        metadata = parse_filename_metadata(filename)

        # Smart initialization
        init = smart_initial_guess(freq, Z_re, Z_im)

        # Set initial parameters
        params_init = get_initial_params(init)

        # Fitting
        result = fit_model_optimized(params_init, freq, Z_re, Z_im)
        omega = 2 * np.pi * freq
        Z_fit = circuit_model2(result.x, omega)
        metrics = calculate_metrics(Z_fit, Z_re, Z_im, freq)

        # Prepare results
        param_names = ['Rs', 'R1', 'Q1', 'a1', 'R2', 'Q2', 'a2', 'sigma', 'A', 'B']
        result_dict = {
            'filename': filename,
            'full_path': file_path,
            'status': 'success',
            'mape_re': metrics['mape_re'],
            'mape_im': metrics['mape_im'],
            'mape_re_lf': metrics['mape_re_lf'],
            'mape_im_lf': metrics['mape_im_lf'],
            'r2_re': metrics['r2_re'],
            'r2_im': metrics['r2_im']
        }

        # Add fitting parameters
        for name, val in zip(param_names, result.x):
            result_dict[name] = val

        # Add metadata
        if metadata:
            result_dict['battery_id'] = metadata['battery_id']
            result_dict['test_cycle'] = metadata['test_cycle']
            result_dict['soc_pct'] = metadata['soc_pct']
            result_dict['rest_hours'] = metadata['rest_hours']
            result_dict['battery_label'] = metadata['battery_label']
            result_dict['test_label'] = metadata['test_label']

        # Plot individual figures (if needed)
        if plot_individual:
            base_name = os.path.splitext(filename)[0]
            fig_path = os.path.join(output_dir, f"EIS_Model2_{base_name}.png")
            plot_results(freq, Z_re, Z_im, Z_fit, metrics, filename, save_path=fig_path)
            plt.close(fig_path)

        print(f"  OK: {filename} (MAPE: {metrics['mape_re']:.1f}%, {metrics['mape_im']:.1f}%)")

        if return_data:
            return result_dict, (freq, Z_re, Z_im, Z_fit)
        else:
            return result_dict

    except Exception as e:
        print(f"  FAIL: {os.path.basename(file_path)} - {str(e)}")
        error_dict = {
            'filename': os.path.basename(file_path),
            'full_path': file_path,
            'status': 'failed',
            'error': str(e)
        }
        if return_data:
            return error_dict, None
        else:
            return error_dict

def batch_fit_folder(folder_path, output_dir=None, plot_individual=False, plot_comparison=False):
    """Batch fit all EIS files in a folder

    Args:
        folder_path: folder path
        output_dir: output directory
        plot_individual: whether to plot individual figures for each file
        plot_comparison: whether to generate comparison plots
    """
    print("="*80)
    print(" Model 2: Batch Fitting Mode")
    print("="*80)

    if not os.path.exists(folder_path):
        print(f"\nError: Folder does not exist {folder_path}")
        return None

    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR

    # Find all txt files
    txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    txt_files = sorted(txt_files)

    if len(txt_files) == 0:
        print(f"\nWarning: No txt files found in {folder_path}")
        return None

    print(f"\nFound {len(txt_files)} EIS files")
    print(f"Folder: {folder_path}")
    print(f"\nStarting batch fitting...\n")

    # Batch processing
    results = []
    results_for_plot = []  # Data for comparison plots

    for i, filename in enumerate(txt_files, 1):
        file_path = os.path.join(folder_path, filename)
        print(f"[{i}/{len(txt_files)}] {filename}...", end=" ")

        # Fit file
        if plot_comparison:
            result, data = fit_single_file(file_path, output_dir, plot_individual, return_data=True)
            results.append(result)
            results_for_plot.append((result, data))
        else:
            result = fit_single_file(file_path, output_dir, plot_individual, return_data=False)
            results.append(result)

    # Save results to CSV
    csv_path = os.path.join(output_dir, "EIS_Model2_Batch_Results.csv")

    # Collect all column names
    all_columns = set()
    for r in results:
        all_columns.update(r.keys())

    # Sort columns and ensure certain columns come first
    priority_cols = ['filename', 'battery_label', 'test_label', 'soc_pct', 'rest_hours',
                     'status', 'mape_re', 'mape_im', 'mape_re_lf', 'mape_im_lf']
    priority_cols = [c for c in priority_cols if c in all_columns]
    other_cols = sorted([c for c in all_columns if c not in priority_cols])
    final_columns = priority_cols + other_cols

    # Write to CSV
    df = pd.DataFrame(results)
    df = df[final_columns]
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')

    print(f"\n{'='*80}")
    print(f"Batch fitting completed!")
    print(f"{'='*80}")
    print(f"\nResult statistics:")
    print(f"  Total files: {len(results)}")
    print(f"  Success: {sum(1 for r in results if r['status'] == 'success')}")
    print(f"  Failed: {sum(1 for r in results if r['status'] == 'failed')}")

    if sum(1 for r in results if r['status'] == 'success') > 0:
        success_results = [r for r in results if r['status'] == 'success']
        print(f"\nFitting quality (successful files):")
        print(f"  Average MAPE real: {np.mean([r['mape_re'] for r in success_results]):.2f}%")
        print(f"  Average MAPE imaginary: {np.mean([r['mape_im'] for r in success_results]):.2f}%")
        print(f"  Average low frequency MAPE real: {np.mean([r['mape_re_lf'] for r in success_results]):.2f}%")
        print(f"  Average low frequency MAPE imaginary: {np.mean([r['mape_im_lf'] for r in success_results]):.2f}%")

    print(f"\nResults saved to: {csv_path}")

    # Generate comparison plot (if needed)
    if plot_comparison:
        print(f"\n{'='*80}")
        print("Generating comparison plot...")
        print(f"{'='*80}")
        folder_name = os.path.basename(folder_path)
        plot_comparison_nyquist(results_for_plot, output_dir, folder_name)

    return df

def main(file_path):
    """Main function - single file mode"""
    print("="*80)
    print(" Model 2: Dual RC + Bounded Warburg - Universal EIS Fitting Tool")
    print("="*80)

    # Read data
    print(f"\nReading EIS data from: {file_path}")
    freq, Z_re, Z_im = read_eis_data(file_path, filter_negative_im=True)

    filename = os.path.basename(file_path)
    print(f"\nData info:")
    print(f"  File: {filename}")
    print(f"  Data points: {len(freq)}")
    print(f"  Freq range: {np.min(freq):.3f} - {np.max(freq):.1f} Hz")
    print(f"  Low freq points (<1Hz): {np.sum(freq < 1.0)}")

    # Parse metadata
    metadata = parse_filename_metadata(filename)
    if metadata:
        print(f"\nFile metadata:")
        print(f"  Battery: {metadata['battery_label']}")
        print(f"  Test: {metadata['test_label']}")
        print(f"  SOC: {metadata['soc_pct']}%")
        print(f"  Rest time: {metadata['rest_hours']}h")

    # Smart initialization
    init = smart_initial_guess(freq, Z_re, Z_im)
    print(f"\nSmart initialization:")
    print(f"  Rs = {init['Rs']:.6e}")
    print(f"  R_total = {init['R_total']:.6e}")

    # Set initial parameters
    params_init = get_initial_params(init)

    # Fitting
    print(f"\n{'='*80}")
    print("Fitting Model 2...")
    print(f"{'='*80}")

    result = fit_model_optimized(params_init, freq, Z_re, Z_im)
    omega = 2 * np.pi * freq
    Z_fit = circuit_model2(result.x, omega)
    metrics = calculate_metrics(Z_fit, Z_re, Z_im, freq)

    # Print parameters
    param_names = ['Rs', 'R1', 'Q1', 'a1', 'R2', 'Q2', 'a2', 'sigma', 'A', 'B']
    print(f"\nFitted parameters:")
    for name, val in zip(param_names, result.x):
        print(f"  {name:8s} = {val:.6e}")

    # Print fitting quality
    print(f"\nOverall metrics:")
    print(f"  MAPE:  Real={metrics['mape_re']:.2f}%, Imag={metrics['mape_im']:.2f}%")
    print(f"  R2:    Real={metrics['r2_re']:.6f}, Imag={metrics['r2_im']:.6f}")

    print(f"\nLow freq metrics (<1Hz):")
    print(f"  MAPE:  Real={metrics['mape_re_lf']:.2f}%, Imag={metrics['mape_im_lf']:.2f}%")

    # Generate figure filename
    output_dir = DEFAULT_OUTPUT_DIR
    base_name = os.path.splitext(filename)[0]
    fig_path = os.path.join(output_dir, f"EIS_Model2_{base_name}.png")
    csv_path = os.path.join(output_dir, "EIS_Model2_Parameters.csv")

    # Plotting
    fig = plot_results(freq, Z_re, Z_im, Z_fit, metrics, filename, save_path=fig_path)

    # Save parameters to CSV
    save_metadata = {
        'filename': filename,
        'mape_re': metrics['mape_re'],
        'mape_im': metrics['mape_im'],
        'mape_re_lf': metrics['mape_re_lf'],
        'mape_im_lf': metrics['mape_im_lf']
    }
    if metadata:
        save_metadata.update(metadata)

    save_parameters_to_csv(result.x, param_names, csv_path, metadata=save_metadata)

    print(f"\n{'='*80}")
    print("Fitting completed successfully!")
    print(f"{'='*80}")

    return result, metrics

if __name__ == "__main__":

    # ===========================
    # Determine run mode
    # ===========================
    # Single file fitting
    file_path = r"E:\Datasets\314Ah\Bat4\RPT1\EIS对比\#4-RPT1-D75%-16h.txt"
    # Batch fitting (folder)
    # batch_folder = r"E:\Datasets\314Ah\Bat4\RPT1\10h"  # Modify this to your folder path

    # Batch mode: folder path configured in script
    if 'batch_folder' in dir() and os.path.exists(batch_folder):
        df = batch_fit_folder(batch_folder, plot_individual=True, plot_comparison=True)
        # plot_individual: whether to generate figures; plot_comparison: whether to generate comparison plots

    # Single file mode: command line arguments or script configuration
    else:
        # Get file path from command line arguments
        if len(sys.argv) > 1:
            file_path = sys.argv[1]
        # Use file path configured in script
        else:
            file_path = file_path

        try:
            result, metrics = main(file_path)
            plt.show()

        except FileNotFoundError as e:
            print(f"\nError: {e}")
            print("\nUsage:")
            print("  Single file fitting:")
            print("    python EIS_Model2_Universal.py \"file_path.txt\"")
            print("    Or modify the file_path variable at the top of the script")
            print("\n  Batch fitting:")
            print("    Modify the batch_folder variable at the top of the script")

        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()
