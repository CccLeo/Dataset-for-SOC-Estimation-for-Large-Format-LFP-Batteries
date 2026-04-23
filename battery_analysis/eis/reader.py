"""EIS data reading module"""

import pandas as pd
import os

# Low frequency threshold (Hz)
LOW_FREQ_THRESHOLD = 1.0


def read_eis_data(file_path, skiprows=1, filter_negative_im=True):
    """Read EIS data

    Args:
        file_path: file path
        skiprows: number of rows to skip
        filter_negative_im: whether to filter negative imaginary part data

    Returns:
        freq, Z_re, Z_im: frequency, real part, imaginary part arrays
    """
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
