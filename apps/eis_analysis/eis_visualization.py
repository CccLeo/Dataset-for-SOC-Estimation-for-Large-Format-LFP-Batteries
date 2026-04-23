"""EIS Data Plotting Script
Supports single file plotting and batch comparison plotting
Supports plotting of real part, imaginary part vs SOC relationships
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import warnings
warnings.filterwarnings('ignore')

def plot_single_eis(file_path, ax=None, show_plot=True, save_path=None,
                    mark_freq=None, show_freq_labels=False):
    """Plot Nyquist plot for a single EIS file

    Args:
        file_path: EIS data file path
        ax: matplotlib axes object, if None create new figure
        show_plot: whether to display the plot
        save_path: save path, if None do not save
        mark_freq: list of frequencies to mark (unit: Hz), e.g. [1000, 100, 10, 1, 0.1, 0.01]
        show_freq_labels: whether to display frequency labels on the plot

    Returns:
        fig: matplotlib figure object
    """
    # Read EIS data
    data = pd.read_csv(file_path, skiprows=1, delimiter='\t')

    # Extract data columns
    freq = data['Freq(Hz)']
    Z_re = data["Z'(Ohm.cm²)"]
    Z_im = data["Z''(Ohm.cm²)"]

    # Only keep data points with negative imaginary part
    mask = Z_im < 0
    freq = freq[mask]
    Z_re = Z_re[mask]
    Z_im = Z_im[mask]

    # If ax is not provided, create new figure
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig = ax.figure

    # Generate legend label (using filename)
    filename = os.path.splitext(os.path.basename(file_path))[0]
    ax.plot(Z_re, -Z_im, 'o-', label=filename, markersize=6, linewidth=1.5, alpha=0.8)

    # Mark special frequency points
    if show_freq_labels and mark_freq is not None:
        # Collect points to mark
        marked_x = []
        marked_y = []
        marked_labels = []

        for target_freq in mark_freq:
            # Find the data point closest to the target frequency
            idx = (np.abs(freq.values - target_freq)).argmin()
            actual_freq = freq.values[idx]

            # Only label points with similar frequency (error within 50%)
            if abs(actual_freq - target_freq) / target_freq < 0.5:
                marked_x.append(Z_re.values[idx])
                marked_y.append(-Z_im.values[idx])

                # Choose appropriate label format based on frequency magnitude
                if actual_freq >= 1000:
                    label = f'{actual_freq/1000:.1f} kHz'
                elif actual_freq >= 1:
                    label = f'{actual_freq:.0f} Hz'
                else:
                    label = f'{actual_freq:.3f} Hz'
                marked_labels.append(label)

        # Mark special frequency points with solid dots
        if marked_x:
            ax.scatter(marked_x, marked_y, s=60, c='red', marker='o',
                      edgecolors='darkred', linewidths=1, zorder=5,
                      label='Marked Freq' if ax is None else '')

            # Add frequency labels, uniformly placed in the upper left direction of the marked points
            for x, y, label in zip(marked_x, marked_y, marked_labels):
                ax.annotate(label, xy=(x, y), xytext=(-10, 15),
                           textcoords='offset points',
                           fontsize=8, color='black',
                           ha='right', va='bottom',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                  alpha=0.85, edgecolor='gray', linewidth=0.5))

    # Set axis labels
    ax.set_xlabel("Z' (Ohm·cm²)", fontsize=12, fontweight='bold')
    ax.set_ylabel("-Z'' (Ohm·cm²)", fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')

    # If independent plotting, set title and legend
    if ax is None:
        ax.set_title(f'EIS Nyquist Plot\n{filename}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")

        if show_plot:
            plt.show()

    return fig


def plot_batch_eis(folder_path, pattern='*2h*.txt', show_plot=True, save_path=None,
                  mark_freq=None, show_freq_labels=False):
    """Batch plot Nyquist comparison plots for multiple EIS files

    Args:
        folder_path: EIS data folder path
        pattern: file matching pattern, default '*2h*.txt'
        show_plot: whether to display the plot
        save_path: save path, if None do not save
        mark_freq: list of frequencies to mark (unit: Hz), e.g. [1000, 100, 10, 1, 0.1, 0.01]
        show_freq_labels: whether to display frequency labels on the plot

    Returns:
        fig: matplotlib figure object
    """
    # Find matching files
    search_pattern = os.path.join(folder_path, pattern)
    files = sorted(glob.glob(search_pattern))

    if not files:
        print(f"No matching files found: {search_pattern}")
        return None

    print(f"Found {len(files)} files:")
    for f in files:
        print(f"  - {os.path.basename(f)}")

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))

    # Define color cycle
    colors = plt.cm.viridis(np.linspace(0, 1, len(files)))

    # Plot each file
    import re
    for i, file_path in enumerate(files):
        data = pd.read_csv(file_path, skiprows=1, delimiter='\t')

        freq = data['Freq(Hz)']
        Z_re = data["Z'(Ohm.cm²)"]
        Z_im = data["Z''(Ohm.cm²)"]

        # Only keep data points with negative imaginary part
        mask = Z_im < 0
        freq = freq[mask]
        Z_re = Z_re[mask]
        Z_im = Z_im[mask]

        # Extract SOC information from filename to generate legend label (e.g. D25 -> 25%)
        basename = os.path.basename(file_path)
        soc_match = re.search(r"D(\d+)", basename)
        if soc_match:
            soc_val = int(soc_match.group(1))
            label = f"{soc_val}%"
        else:
            # If SOC pattern not found, use filename
            label = os.path.splitext(basename)[0]

        ax.plot(Z_re, -Z_im, 'o-', label=label, markersize=6,
                linewidth=1.5, color=colors[i], alpha=0.8)

        # Mark special frequency points (mark for each curve)
        if show_freq_labels and mark_freq is not None:
            # Collect points to mark
            marked_x = []
            marked_y = []
            marked_labels = []

            for target_freq in mark_freq:
                # Find the data point closest to the target frequency
                idx = (np.abs(freq.values - target_freq)).argmin()
                actual_freq = freq.values[idx]

                # Only label points with similar frequency (error within 50%)
                if abs(actual_freq - target_freq) / target_freq < 0.5:
                    marked_x.append(Z_re.values[idx])
                    marked_y.append(-Z_im.values[idx])

                    # Choose appropriate label format based on frequency magnitude
                    if actual_freq >= 1000:
                        label_freq = f'{actual_freq/1000:.1f} kHz'
                    elif actual_freq >= 1:
                        label_freq = f'{actual_freq:.0f} Hz'
                    else:
                        label_freq = f'{actual_freq:.3f} Hz'
                    marked_labels.append(label_freq)

            # Mark special frequency points with solid dots (use corresponding curve color)
            if marked_x:
                ax.scatter(marked_x, marked_y, s=60, c=colors[i], marker='o',
                          edgecolors='black', linewidths=0.8, zorder=5)

                # Only add labels for the first curve to avoid overlapping
                if i == 0:
                    for x, y, label_freq in zip(marked_x, marked_y, marked_labels):
                        # Uniformly placed in the upper left direction of the marked points
                        ax.annotate(label_freq, xy=(x, y), xytext=(-10, 15),
                                   textcoords='offset points',
                                   fontsize=10, color='black',
                                   ha='right', va='bottom',
                                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                          alpha=0.85, edgecolor='gray', linewidth=0.8))

    # Set axis labels
    ax.set_xlabel("Z' (Ohm·cm²)", fontsize=12, fontweight='bold')
    ax.set_ylabel("-Z'' (Ohm·cm²)", fontsize=12, fontweight='bold')
    # ax.set_title('EIS Nyquist Plot Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12, loc='best')
    ax.axis('equal')

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")

    if show_plot:
        plt.show()

    return fig


def plot_batch_eis_multiple_folders(folder_paths, pattern='*2h*.txt', show_plot=True,
                                   save_path=None, mark_freq=None, show_freq_labels=False):
    """Batch plot Nyquist comparison plots for EIS files from multiple folders

    Args:
        folder_paths: list of folder paths
        pattern: file matching pattern, default '*2h*.txt'
        show_plot: whether to display the plot
        save_path: save path, if None do not save
        mark_freq: list of frequencies to mark (unit: Hz), e.g. [1000, 100, 10, 1, 0.1, 0.01]
        show_freq_labels: whether to display frequency labels on the plot

    Returns:
        fig: matplotlib figure object
    """
    # Collect all files
    all_files = []
    for folder_path in folder_paths:
        search_pattern = os.path.join(folder_path, pattern)
        files = sorted(glob.glob(search_pattern))
        all_files.extend(files)

    if not all_files:
        print(f"No matching files found")
        return None

    print(f"Found {len(all_files)} files:")
    for f in all_files:
        print(f"  - {os.path.basename(f)}")

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))

    # Define color cycle
    colors = plt.cm.viridis(np.linspace(0, 1, len(all_files)))

    # Plot each file
    import re
    for i, file_path in enumerate(all_files):
        data = pd.read_csv(file_path, skiprows=1, delimiter='\t')

        freq = data['Freq(Hz)']
        Z_re = data["Z'(Ohm.cm²)"]
        Z_im = data["Z''(Ohm.cm²)"]

        # Only keep data points with negative imaginary part
        mask = Z_im < 0
        freq = freq[mask]
        Z_re = Z_re[mask]
        Z_im = Z_im[mask]

        # Extract SOC information from filename to generate legend label (e.g. D25 -> 25%)
        basename = os.path.basename(file_path)
        soc_match = re.search(r"D(\d+)", basename)
        if soc_match:
            soc_val = int(soc_match.group(1))
            label = f"{soc_val}%"
        else:
            # If SOC pattern not found, keep folder + filename
            folder_name = os.path.basename(os.path.dirname(file_path))
            filename = os.path.splitext(basename)[0]
            label = f"{folder_name}/{filename}"

        ax.plot(Z_re, -Z_im, 'o-', label=label, markersize=6,
                linewidth=1.5, color=colors[i], alpha=0.8)

        # Mark special frequency points (mark for each curve)
        if show_freq_labels and mark_freq is not None:
            # Collect points to mark
            marked_x = []
            marked_y = []
            marked_labels = []

            for target_freq in mark_freq:
                # Find the data point closest to the target frequency
                idx = (np.abs(freq.values - target_freq)).argmin()
                actual_freq = freq.values[idx]

                # Only label points with similar frequency (error within 50%)
                if abs(actual_freq - target_freq) / target_freq < 0.5:
                    marked_x.append(Z_re.values[idx])
                    marked_y.append(-Z_im.values[idx])

                    # Choose appropriate label format based on frequency magnitude
                    if actual_freq >= 1000:
                        label_freq = f'{actual_freq/1000:.1f} kHz'
                    elif actual_freq >= 1:
                        label_freq = f'{actual_freq:.0f} Hz'
                    else:
                        label_freq = f'{actual_freq:.3f} Hz'
                    marked_labels.append(label_freq)

            # Mark special frequency points with solid dots (use corresponding curve color)
            if marked_x:
                ax.scatter(marked_x, marked_y, s=60, c=colors[i], marker='o',
                          edgecolors='black', linewidths=0.8, zorder=5)

                # Only add labels for the first curve to avoid overlapping
                if i == 0:
                    for x, y, label_freq in zip(marked_x, marked_y, marked_labels):
                        # Uniformly placed in the upper left direction of the marked points
                        ax.annotate(label_freq, xy=(x, y), xytext=(-10, 15),
                                   textcoords='offset points',
                                   fontsize=8, color='black',
                                   ha='right', va='bottom',
                                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                          alpha=0.85, edgecolor='gray', linewidth=0.5))

    # Set axis labels
    ax.set_xlabel("Z' (Ohm·cm²)", fontsize=12, fontweight='bold')
    ax.set_ylabel("-Z'' (Ohm·cm²)", fontsize=12, fontweight='bold')
    ax.set_title('EIS Nyquist Plot Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc='best')
    ax.axis('equal')

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")

    if show_plot:
        plt.show()

    return fig


def plot_eis_temperature_comparison(folder_paths, temperature_labels, pattern='*D*1h*.txt',
                                   show_plot=True, save_path=None, mark_freq=None, show_freq_labels=False):
    """
    Plot four subplots to compare EIS curves at different temperatures, 2x2 layout

    Args:
        folder_paths: list of four folder paths, in order 15°C, 20°C, 25°C, 30°C
        temperature_labels: list of four temperature labels, e.g. ['15°C', '20°C', '25°C', '30°C']
        pattern: file matching pattern, default '*D*1h*.txt'
        show_plot: whether to display the plot
        save_path: plot save path, if None do not save
        mark_freq: list of frequencies to mark (unit: Hz), e.g. [1000, 100, 10, 1, 0.1, 0.01]
        show_freq_labels: whether to display frequency labels on the plot

    Returns:
        fig: matplotlib figure object
    """
    import re

    # Set global font and plotting style (academic paper style, no bold)
    plt.rcParams['font.family'] = 'Calibri'
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.linewidth'] = 1.0
    plt.rcParams['xtick.major.width'] = 1.0
    plt.rcParams['ytick.major.width'] = 1.0
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['mathtext.default'] = 'regular'
    plt.rcParams['font.weight'] = 'normal'
    plt.rcParams['axes.labelweight'] = 'normal'
    plt.rcParams['axes.titleweight'] = 'normal'

    # Step 1: First collect SOC values from all files, create global unified color mapping
    all_soc_values = []
    for folder_path in folder_paths:
        search_pattern = os.path.join(folder_path, pattern)
        files = sorted(glob.glob(search_pattern))
        for file_path in files:
            basename = os.path.basename(file_path)
            soc_match = re.search(r"D(\d+)", basename)
            if soc_match:
                soc_val = int(soc_match.group(1))
                all_soc_values.append(soc_val)

    if not all_soc_values:
        print("No valid SOC data found")
        return None

    # Create global color mapping (four subplots unified SOC corresponds to same color)
    soc_min, soc_max = min(all_soc_values), max(all_soc_values)
    norm = plt.Normalize(soc_min, soc_max)
    cmap = plt.cm.viridis

    # Create 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))  # Increase a bit of width for colorbar
    axes = axes.flatten()  # Flatten to 1D array for easy indexing
    for ax in axes:
        ax.set_box_aspect(1)
    # Subplot labels (a), (b), (c), (d)
    subplot_labels = ['(a)', '(b)', '(c)', '(d)']

    for idx, (folder_path, temp_label, ax) in enumerate(zip(folder_paths, temperature_labels, axes)):
        # Find matching files
        search_pattern = os.path.join(folder_path, pattern)
        files = sorted(glob.glob(search_pattern))

        if not files:
            print(f"Temperature {temp_label}: No matching files found: {search_pattern}")
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
            continue

        print(f"\nTemperature {temp_label}: Found {len(files)} files")
        for f in files:
            print(f"  - {os.path.basename(f)}")

        # Plot each file
        for i, file_path in enumerate(files):
            data = pd.read_csv(file_path, skiprows=1, delimiter='\t')

            freq = data['Freq(Hz)']
            Z_re = data["Z'(Ohm.cm²)"]
            Z_im = data["Z''(Ohm.cm²)"]

            # Only keep data points with negative imaginary part
            mask = Z_im < 0
            freq = freq[mask]
            Z_re = Z_re[mask]
            Z_im = Z_im[mask]

            # Extract SOC information from filename to generate legend label (e.g. D25 -> 25%)
            basename = os.path.basename(file_path)
            soc_match = re.search(r"D(\d+)", basename)
            if soc_match:
                soc_val = int(soc_match.group(1))
                label = f"{soc_val}%"
            else:
                # If SOC pattern not found, use filename
                label = os.path.splitext(basename)[0]
                soc_val = 0  # Default value

            # Get color from global color mapping based on SOC value (four subplots same SOC same color)
            color = cmap(norm(soc_val))

            ax.plot(Z_re, -Z_im, 'o-', label=label, markersize=5,
                    linewidth=1.2, color=color, alpha=0.8)

            # Mark special frequency points (mark for each curve)
            if show_freq_labels and mark_freq is not None:
                # Collect points to mark
                marked_x = []
                marked_y = []
                marked_labels = []

                for target_freq in mark_freq:
                    # Find the data point closest to the target frequency
                    idx_freq = (np.abs(freq.values - target_freq)).argmin()
                    actual_freq = freq.values[idx_freq]

                    # Only label points with similar frequency (error within 50%)
                    if abs(actual_freq - target_freq) / target_freq < 0.5:
                        marked_x.append(Z_re.values[idx_freq])
                        marked_y.append(-Z_im.values[idx_freq])

                        # Choose appropriate label format based on frequency magnitude
                        if actual_freq >= 1000:
                            label_freq = f'{actual_freq/1000:.1f} kHz'
                        elif actual_freq >= 1:
                            label_freq = f'{actual_freq:.0f} Hz'
                        else:
                            label_freq = f'{actual_freq:.3f} Hz'
                        marked_labels.append(label_freq)

                # Mark special frequency points with solid dots (use corresponding curve color)
                if marked_x:
                    ax.scatter(marked_x, marked_y, s=50, c=color, marker='o',
                              edgecolors='black', linewidths=0.8, zorder=5)

                    # First curve in each subplot displays frequency labels
                    if i == 0:
                        for x, y, label_freq in zip(marked_x, marked_y, marked_labels):
                            # Uniformly placed in the upper left direction of the marked points
                            ax.annotate(label_freq, xy=(x, y), xytext=(-10, 6),
                                       textcoords='offset points',
                                       fontsize=9, color='black',
                                       ha='right', va='bottom',
                                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                              alpha=0.85, edgecolor='gray', linewidth=0.8))

        # Set subplot properties
        ax.set_xlabel("Z' (Ω·cm²)", fontsize=12)
        ax.set_ylabel("-Z'' (Ω·cm²)", fontsize=12)
        ax.grid(True, alpha=0.3)
        # ax.legend(fontsize=10, loc='best')
        ax.axis('equal')

        # Add subplot label + temperature title centered below the subplot
        ax.text(0.5, -0.16, f'{subplot_labels[idx]} EIS test at {temp_label}', ha='center',
                transform=ax.transAxes, fontsize=14)

    # Add global colorbar (corresponds to SOC values)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Only for initialization, no actual data needed
    cbar = fig.colorbar(
        sm,
        ax=axes,
        location='right',
        pad=0.15,
        fraction=0.02,
        aspect=50,  # Increase aspect value: height/width ratio of vertical colorbar, larger value makes colorbar longer and thinner
        shrink=1.0  # Scaling ratio, >1 zooms in and stretches, <1 zooms out
    )
    cbar.set_label('SOC (%)', fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    # Adjust subplot spacing: increase right to expand subplot area to the right, reduce wasted white space on the right
    plt.tight_layout(rect=[0, 0.02, 0.85, 0.98], pad=1.0, w_pad=1.0, h_pad=1.0)

    # Save plot
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nComparison plot saved to: {save_path}")

    # Display plot
    if show_plot:
        plt.show()

    return fig


def plot_impedance_vs_soc(folder_path, pattern='*1h*.txt', target_freq_re=0.01, target_freq_im=0.1,
                        show_plot=True, save_path=None, data_save_path=None):
    """
    Plot the relationship between real/imaginary parts and SOC at specific frequencies
    Real part uses frequency specified by target_freq_re, imaginary part uses frequency specified by target_freq_im

    Args:
        folder_path: EIS data folder path
        pattern: file matching pattern, default '*1h*.txt'
        target_freq_re: target frequency for real part (Hz), default 0.01
        target_freq_im: target frequency for imaginary part (Hz), default 0.1
        show_plot: whether to display the plot
        save_path: plot save path, if None do not save plot
        data_save_path: data save path (csv format), if None do not save data

    Returns:
        fig: matplotlib figure object
    """
    import re

    # Find matching files
    search_pattern = os.path.join(folder_path, pattern)
    files = sorted(glob.glob(search_pattern))

    if not files:
        print(f"No matching files found: {search_pattern}")
        return None

    print(f"Found {len(files)} files:")
    for f in files:
        print(f"  - {os.path.basename(f)}")

    # Collect data
    soc_values = []
    Z_re_values = []
    Z_im_values = []
    actual_freq_re_list = []
    actual_freq_im_list = []

    for file_path in files:
        try:
            # Read EIS data
            data = pd.read_csv(file_path, skiprows=1, delimiter='\t')

            freq = data['Freq(Hz)']
            Z_re = data["Z'(Ohm.cm²)"]
            Z_im = data["Z''(Ohm.cm²)"]

            # Find the data point closest to the real part target frequency
            idx_re = (np.abs(freq.values - target_freq_re)).argmin()
            actual_freq_re = freq.values[idx_re]

            # Check real part frequency error (allow 50% error)
            if abs(actual_freq_re - target_freq_re) / target_freq_re > 0.5:
                print(f"  {os.path.basename(file_path)}: No real part frequency point close to {target_freq_re} Hz found (actual: {actual_freq_re:.4f} Hz)")
                continue

            # Find the data point closest to the imaginary part target frequency
            idx_im = (np.abs(freq.values - target_freq_im)).argmin()
            actual_freq_im = freq.values[idx_im]

            # Check imaginary part frequency error (allow 50% error)
            if abs(actual_freq_im - target_freq_im) / target_freq_im > 0.5:
                print(f"  {os.path.basename(file_path)}: No imaginary part frequency point close to {target_freq_im} Hz found (actual: {actual_freq_im:.4f} Hz)")
                continue

            # Extract SOC value from filename (e.g. #1-RPT2-D25-1h.txt -> 25)
            basename = os.path.basename(file_path)
            soc_match = re.search(r"D(\d+)", basename)
            if soc_match:
                soc = float(soc_match.group(1))
            else:
                print(f"  {os.path.basename(file_path)}: Cannot extract SOC value from filename, skipping")
                continue

            # Store data
            soc_values.append(soc)
            Z_re_values.append(Z_re.values[idx_re])
            Z_im_values.append(Z_im.values[idx_im])
            actual_freq_re_list.append(actual_freq_re)
            actual_freq_im_list.append(actual_freq_im)

            print(f"  SOC {soc}%:")
            print(f"    Real part @ {actual_freq_re:.3f} Hz: Z'={Z_re.values[idx_re]:.7f} Ohm·cm²")
            print(f"    Imaginary part @ {actual_freq_im:.3f} Hz: Z''={Z_im.values[idx_im]:.7f} Ohm·cm²")

        except Exception as e:
            print(f"  {os.path.basename(file_path)}: Processing failed: {e}")
            continue

    if not soc_values:
        print("No valid data")
        return None

    # Sort by SOC
    sorted_indices = np.argsort(soc_values)
    soc_values = [soc_values[i] for i in sorted_indices]
    Z_re_values = [Z_re_values[i] for i in sorted_indices]
    Z_im_values = [Z_im_values[i] for i in sorted_indices]
    actual_freq_re_list = [actual_freq_re_list[i] for i in sorted_indices]
    actual_freq_im_list = [actual_freq_im_list[i] for i in sorted_indices]

    # Set global font to Calibri
    plt.rcParams['font.family'] = 'Calibri'
    plt.rcParams['font.size'] = 11


    # Create figure (two subplots)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Subplot 1: Real part vs SOC
    ax1.plot(soc_values, Z_re_values, 'o-', markersize=8, linewidth=2, color='blue', alpha=0.8)
    ax1.set_xlabel('SOC (%)', fontsize=11)
    ax1.set_ylabel(f"Z' (Ohm·cm²)", fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='both', which='major', labelsize=11)
    # Add numbered title directly below the subplot
    ax1.text(0.5, -0.1, f'(a) Real Part vs SOC (@{target_freq_re} Hz)',
             transform=ax1.transAxes, ha='center', va='top', fontsize=14)

    # Subplot 2: Imaginary part vs SOC
    ax2.plot(soc_values, Z_im_values, 's-', markersize=8, linewidth=2, color='red', alpha=0.8)
    ax2.set_xlabel('SOC (%)', fontsize=11)
    ax2.set_ylabel(f"Z'' (Ohm·cm²)", fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='both', which='major', labelsize=11)
    # Add numbered title directly below the subplot
    ax2.text(0.5, -0.1, f'(b) Imaginary Part vs SOC (@{target_freq_im} Hz)',
             transform=ax2.transAxes, ha='center', va='top', fontsize=14)

    # Adjust layout to make space for bottom titles
    plt.tight_layout(rect=[0, 0.01, 1, 1])

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")

    # Save data to CSV file (do not save header)
    if data_save_path:
        os.makedirs(os.path.dirname(data_save_path) or '.', exist_ok=True)
        # Write data manually, do not save header, use fixed decimal format to avoid scientific notation
        with open(data_save_path, 'w') as f:
            for soc, z_re, z_im, freq_re, freq_im in zip(soc_values, Z_re_values, Z_im_values, actual_freq_re_list, actual_freq_im_list):
                # Use sufficient decimal places to avoid scientific notation
                f.write(f"{soc:.1f},{z_re:.10f},{z_im:.10f},{freq_re:.6f},{freq_im:.6f}\n")
        print(f"\nData saved to: {data_save_path}")
        print(f"CSV format: SOC(%), Z'_re(Ohm·cm²), Z''_im(Ohm·cm²), actual frequency real part (Hz), actual frequency imaginary part (Hz)")

    if show_plot:
        plt.show()

    return fig


# ==================== 使用示例 ====================

if __name__ == "__main__":

    # 定义常用频率点
    COMMON_FREQ = [1, 0.1, 0.01]

    # 示例1: 绘制四温度EIS对比图（2x2子图）
    # folder_paths = [
    #     r"E:\Datasets\314Ah\Bat2\EIS4SOC\15degC\无间隔",
    #     r"E:\Datasets\314Ah\Bat2\EIS4SOC\20degC\无间隔",
    #     r"E:\Datasets\314Ah\Bat2\EIS4SOC\25degC\无间隔1hEIS测试\2nd",
    #     r"E:\Datasets\314Ah\Bat2\EIS4SOC\30degC"
    # ]
    # temperature_labels = ['15°C', '20°C', '25°C', '30°C']
    #
    # plot_eis_temperature_comparison(
    #     folder_paths,
    #     temperature_labels,
    #     pattern='*D*1h*.txt',
    #     show_plot=True,
    #     save_path=r"E:\PycharmProjects\PythonProject1\output\EIS_Temperature_Comparison.png",
    #     mark_freq=COMMON_FREQ,
    #     show_freq_labels=True
    # )

    # 示例2: 绘制0.01Hz下实部-soc关系和0.1Hz下的虚部-soc关系
    folder_path = r"E:\Datasets\314Ah\Bat2\EIS4SOC\17degC"
    plot_impedance_vs_soc(folder_path, pattern='*D*1h*.txt', target_freq_re=0.01, target_freq_im=0.1,
                       show_plot=True, save_path=None,
    data_save_path=r"E:\PycharmProjects\PythonProject1\output\SOC-EIS-test17.csv")

    # 示例2: 绘制单个文件并标记频率
    # file_path = r"E:\Datasets\314Ah\Bat4\RPT1\EIS对比\#4-RPT1-D75%-16h.txt"
    # plot_single_eis(file_path, show_plot=True, save_path=None,
    #                 mark_freq=COMMON_FREQ, show_freq_labels=True)

    # 示例3: 从多个文件夹批量绘制包含 '2h' 的文件
    # folder_paths = [
    #     r"E:\Datasets\314Ah\Bat2\EIS4SOC\25degC\test",
    #     r"E:\Datasets\314Ah\Bat2\EIS4SOC\25degC\有间隔1hEIS测试"
    # ]
    # plot_batch_eis_multiple_folders(folder_paths, pattern='*D*70*1h*.txt', show_plot=True,
    #                                save_path=None, mark_freq=COMMON_FREQ,
    #                                show_freq_labels=True)

    # 示例4: 单文件夹批量绘制
    # folder_path = r"E:\Datasets\314Ah\Bat2\EIS4SOC\25degC\无间隔1hEIS测试\2nd"
    # plot_batch_eis(folder_path, pattern='*D*1h*.txt', show_plot=True, save_path=None,
    #               mark_freq=COMMON_FREQ, show_freq_labels=True)

    # 示例5: 批量绘制并保存
    # plot_batch_eis_multiple_folders(folder_paths, pattern='*2h*.txt', show_plot=True,
    #                                save_path=r"E:\PycharmProjects\PythonProject1\output\EIS_Batch.png",
    #                                mark_freq=COMMON_FREQ, show_freq_labels=True)
