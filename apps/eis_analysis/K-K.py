import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import lsq_linear
import pandas as pd
import sys
sys.path.append("E:/PycharmProjects/Paper1")
import os
import warnings
warnings.filterwarnings('ignore')

from battery_analysis.eis import read_eis_data
file_path = r"E:\Datasets\314Ah\Bat2\EIS4SOC\25degC\无间隔1hEIS测试\2nd\#2-D50-1h.txt"


def lin_kk_analysis(freq, z_real, z_imag, n_tau=50, fit_inductance=True):
    """
    Linear Kramers-Kronig analysis based on RC relaxation elements.

    Parameters
    ----------
    freq : array
        Frequency in Hz.
    z_real : array
        Real part of impedance.
    z_imag : array
        Imaginary part of impedance.
    n_tau : int
        Number of RC relaxation elements.
    fit_inductance : bool
        Whether to include an inductance term at high frequency.

    Returns
    -------
    result : dict
        KK fitting results.
    """

    # freq = np.asarray(freq)
    # z_real = np.asarray(z_real)
    # z_imag = np.asarray(z_imag)

    omega = 2 * np.pi * freq
    z_exp = z_real + 1j * z_imag

    # Sort data by frequency from high to low or low to high.
    # The fitting itself does not require a specific order, but plotting is clearer.
    idx = np.argsort(freq)
    freq = freq[idx]
    omega = omega[idx]
    z_real = z_real[idx]
    z_imag = z_imag[idx]
    z_exp = z_exp[idx]

    # Time constant range
    tau_min = 0.1 / np.max(omega)
    tau_max = 10 / np.min(omega)
    tau = np.logspace(np.log10(tau_min), np.log10(tau_max), n_tau)

    n = len(freq)

    # Matrix for real part
    A_real = np.zeros((n, n_tau + 1))
    A_real[:, 0] = 1.0  # Rs term

    # Matrix for imaginary part
    A_imag = np.zeros((n, n_tau + 1))
    A_imag[:, 0] = 0.0

    for k in range(n_tau):
        wt = omega * tau[k]
        A_real[:, k + 1] = 1 / (1 + wt**2)
        A_imag[:, k + 1] = -wt / (1 + wt**2)

    # Optional inductance term: Z_L = j omega L
    if fit_inductance:
        A_real = np.column_stack([A_real, np.zeros(n)])
        A_imag = np.column_stack([A_imag, omega])

    # Combine real and imaginary parts
    A = np.vstack([A_real, A_imag])
    b = np.concatenate([z_real, z_imag])

    # Solve least squares problem.
    # Rs and Rk are constrained to be non-negative.
    # Inductance is also constrained to be non-negative if included.
    lower_bounds = np.zeros(A.shape[1])
    upper_bounds = np.full(A.shape[1], np.inf)

    sol = lsq_linear(A, b, bounds=(lower_bounds, upper_bounds), method="trf")
    x = sol.x

    Rs = x[0]
    Rk = x[1:1 + n_tau]

    if fit_inductance:
        L = x[-1]
    else:
        L = 0.0

    # Reconstruct impedance
    z_fit = Rs * np.ones_like(omega, dtype=complex)

    for k in range(n_tau):
        z_fit += Rk[k] / (1 + 1j * omega * tau[k])

    if fit_inductance:
        z_fit += 1j * omega * L

    residual_real = z_real - z_fit.real
    residual_imag = z_imag - z_fit.imag

    z_abs = np.abs(z_exp)
    norm_residual_real = residual_real / z_abs
    norm_residual_imag = residual_imag / z_abs

    rmse = np.sqrt(np.mean(np.abs(z_exp - z_fit) ** 2))
    relative_rmse = rmse / np.mean(np.abs(z_exp))

    result = {
        "freq": freq,
        "omega": omega,
        "tau": tau,
        "Rs": Rs,
        "Rk": Rk,
        "L": L,
        "z_exp": z_exp,
        "z_fit": z_fit,
        "residual_real": residual_real,
        "residual_imag": residual_imag,
        "norm_residual_real": norm_residual_real,
        "norm_residual_imag": norm_residual_imag,
        "rmse": rmse,
        "relative_rmse": relative_rmse,
        "success": sol.success,
    }

    return result


# Example usage

freq, z_real, z_imag = read_eis_data(file_path, filter_negative_im=True)
# freq = freq.values
# z_real = Z_re.values
# z_imag = Z_im.values

result = lin_kk_analysis(freq, z_real, z_imag, n_tau=50, fit_inductance=True)

print("Rs =", result["Rs"])
print("L =", result["L"])
print("RMSE =", result["rmse"])
print("Relative RMSE =", result["relative_rmse"])

plt.figure(figsize=(6, 5))

plt.plot(result["z_exp"].real, -result["z_exp"].imag, "o", label="Experimental")
plt.plot(result["z_fit"].real, -result["z_fit"].imag, "-", label="KK fit")

plt.xlabel("Z' / Ohm")
plt.ylabel("-Z'' / Ohm")
plt.legend()
plt.axis("equal")
plt.tight_layout()
plt.show()