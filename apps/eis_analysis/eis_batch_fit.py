"""
Model 2 EIS Fitting Tool - Main Program (Modular Version)
EIS_Model2: Dual RC + Bounded Warburg Model
See Model2_Documentation.md for documentation
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
sys.path.append("E:/PycharmProjects/PythonProject1")
import os
import warnings

# Import core libraries
from battery_analysis import (
    read_eis_data,
    smart_initial_guess,
    get_initial_params,
    circuit_model2,
    fit_model_optimized,
    calculate_metrics,
    plot_results,
    plot_comparison_nyquist,
    parse_filename_metadata,
    save_parameters_to_csv
)

warnings.filterwarnings('ignore')

DEFAULT_OUTPUT_DIR = r"E:\PycharmProjects\PythonProject1\output"

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
    file_path = r"E:\Datasets\314Ah\Bat2\RPT1\EIS\#2-25%-Longrest.txt"
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
            print("    python EIS_Model2_Main.py \"file_path.txt\"")
            print("    Or modify the file_path variable at the top of the script")
            print("\n  Batch fitting:")
            print("    Modify the batch_folder variable at the top of the script")

        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()
