"""DRT (Distribution of Relaxation Times) analysis core module

Contains core algorithms and data processing functions for DRT analysis
"""

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')


class DRTAnalyzerFinal:
    """Final optimized DRT analyzer"""

    def __init__(self, freq, Z_re, Z_im, tau_min=None, tau_max=None, n_tau=200):
        self.freq = np.asarray(freq, dtype=float)
        self.Z_re = np.asarray(Z_re, dtype=float)
        self.Z_im = np.asarray(Z_im, dtype=float)

        # Sort by frequency from high to low
        sort_idx = np.argsort(-self.freq)
        self.freq = self.freq[sort_idx]
        self.Z_re = self.Z_re[sort_idx]
        self.Z_im = self.Z_im[sort_idx]

        self.omega = 2 * np.pi * self.freq

        # Extend tau range
        if tau_min is None:
            tau_min = 1 / (self.freq.max() * 20)
        if tau_max is None:
            tau_max = 20 / self.freq.min()

        self.tau = np.logspace(np.log10(tau_min), np.log10(tau_max), n_tau)
        self.n_tau = n_tau

        self.gamma = None
        self.Z_fit_re = None
        self.Z_fit_im = None
        self.lambda_opt = None

    def _build_kernel_matrix(self):
        """Build kernel matrix"""
        omega_tau = np.outer(self.omega, self.tau)
        K_re = 1.0 / (1.0 + omega_tau**2)
        K_im = omega_tau / (1.0 + omega_tau**2)
        return K_re, K_im

    def _build_regularization_matrix(self, z=2):
        """Build regularization matrix"""
        if z == 1:
            L = np.diff(np.eye(self.n_tau), n=1, axis=0)
        elif z == 2:
            L = np.diff(np.eye(self.n_tau), n=2, axis=0)
        else:
            raise ValueError("Regularization order only supports 1 or 2")

        for i in range(L.shape[0]):
            L[i, :] /= np.sqrt(np.sum(L[i, :]**2))

        return L

    def fit_optimized(self, lambda_reg=None):
        """
        Optimized fitting method

        Improvements:
        1. Use more stable matrix solving
        2. Add non-negative constraint
        3. Automatic optimal lambda search
        """
        K_re, K_im = self._build_kernel_matrix()
        K = np.vstack([K_re, K_im])
        Z = np.concatenate([self.Z_re, self.Z_im])

        L = self._build_regularization_matrix(z=2)
        KTK = K.T @ K
        KTZ = K.T @ Z
        LTL = L.T @ L

        if lambda_reg is None:
            # 自动搜索最优lambda
            print("Searching optimal regularization parameter...")
            best_lambda = 1e-3
            best_rmse = np.inf
            best_gamma = None

            # 使用更精细的搜索范围
            lambda_range = np.logspace(-8, 4, 100)

            for lam in lambda_range:
                try:
                    A = KTK + lam * LTL

                    # 使用SVD求解，更稳定
                    U, s, Vt = np.linalg.svd(A)
                    gamma = Vt.T @ (U.T @ KTZ / s)
                    gamma = np.maximum(gamma, 0)

                    # 计算RMSE
                    Z_fit = K @ gamma
                    rmse = np.sqrt(np.mean((Z - Z_fit)**2))

                    if rmse < best_rmse:
                        best_rmse = rmse
                        best_lambda = lam
                        best_gamma = gamma.copy()

                except:
                    continue

            print(f"Optimal lambda: {best_lambda:.3e}")
            print(f"Corresponding RMSE: {best_rmse*1000:.4f} mOhm*cm^2")
            lambda_reg = best_lambda
            self.gamma = best_gamma
        else:
            # 使用指定的lambda
            A = KTK + lambda_reg * LTL
            gamma = np.linalg.solve(A, KTZ)
            self.gamma = np.maximum(gamma, 0)

        self.lambda_opt = lambda_reg

        # 计算拟合阻抗
        Z_fit = K @ self.gamma
        self.Z_fit_re = Z_fit[:len(self.freq)]
        self.Z_fit_im = Z_fit[len(self.freq):]

        return self.gamma

    def calculate_rmse(self):
        """Calculate RMSE"""
        if self.Z_fit_re is None:
            raise RuntimeError("Please run fit_optimized() first")

        Z_exp = np.concatenate([self.Z_re, self.Z_im])
        Z_fit = np.concatenate([self.Z_fit_re, self.Z_fit_im])
        return np.sqrt(np.mean((Z_exp - Z_fit)**2))

    def calculate_mape(self, threshold=1e-6):
        """
        Calculate MAPE (improved version, ignore points near 0)

        threshold: data points below this threshold are not included in MAPE calculation
        """
        if self.Z_fit_re is None:
            raise RuntimeError("Please run fit_optimized() first")

        Z_exp = np.concatenate([self.Z_re, self.Z_im])
        Z_fit = np.concatenate([self.Z_fit_re, self.Z_fit_im])

        # 只计算数值较大的点的相对误差
        mask = np.abs(Z_exp) > threshold
        if np.sum(mask) > 0:
            mape = np.mean(np.abs((Z_fit[mask] - Z_exp[mask]) / Z_exp[mask])) * 100
        else:
            mape = np.inf

        return mape

    def calculate_r_squared(self):
        """
        Calculate R² (improved version)
        """
        if self.Z_fit_re is None:
            raise RuntimeError("Please run fit_optimized() first")

        # 分别计算实部和虚部的R²
        ss_res_re = np.sum((self.Z_re - self.Z_fit_re)**2)
        ss_tot_re = np.sum((self.Z_re - np.mean(self.Z_re))**2)
        r2_re = 1.0 - (ss_res_re / ss_tot_re) if ss_tot_re > 1e-20 else 0

        ss_res_im = np.sum((self.Z_im - self.Z_fit_im)**2)
        ss_tot_im = np.sum((self.Z_im - np.mean(self.Z_im))**2)
        r2_im = 1.0 - (ss_res_im / ss_tot_im) if ss_tot_im > 1e-20 else 0

        return (r2_re, r2_im)


def read_and_preprocess_eis(file_path):
    """Read and preprocess EIS data"""
    try:
        from ..eis.reader import read_eis_data
        freq, Z_re, Z_im = read_eis_data(file_path, filter_negative_im=False)
    except:
        df = pd.read_csv(file_path, skiprows=1, delimiter=r'\s+')
        freq = df['Freq(Hz)'].to_numpy()
        Z_re = df["Z'(Ohm.cm^2)"].to_numpy()
        Z_im = df["Z''(Ohm.cm^2)"].to_numpy()

    print(f"Original data points: {len(freq)}")

    # Sort by frequency from high to low
    sort_idx = np.argsort(-freq)
    freq = freq[sort_idx]
    Z_re = Z_re[sort_idx]
    Z_im = Z_im[sort_idx]

    # Smart filtering: only use data with significantly negative imaginary part
    phase = np.degrees(np.arctan2(Z_im, Z_re))
    mask = (Z_im < 0) & (phase < -1)

    n_negative = np.sum(mask)
    print(f"Filtered data points: {n_negative}")

    if n_negative < 5:
        raise ValueError(f"Too few data points (only {n_negative} points)")

    freq = freq[mask]
    Z_re = Z_re[mask]
    Z_im = Z_im[mask]

    print(f"Frequency range: {freq.min():.3e} - {freq.max():.3e} Hz")
    print(f"Impedance real part range: {Z_re.min()*1000:.3f} - {Z_re.max()*1000:.3f} mOhm*cm^2")
    print(f"Impedance imaginary part range: {Z_im.min()*1000:.3f} - {Z_im.max()*1000:.3f} mOhm*cm^2")

    # Estimate Rs
    Rs = Z_re[0]
    print(f"\nEstimated Rs: {Rs*1000:.3f} mOhm*cm^2")

    # Remove Rs
    Z_re_corrected = Z_re - Rs
    print(f"After Rs removal: {Z_re_corrected.min()*1000:.3f} - {Z_re_corrected.max()*1000:.3f} mOhm*cm^2")

    # DRT imaginary part
    Z_im_drt = -Z_im
    print(f"DRT imaginary range: {Z_im_drt.min()*1000:.3f} - {Z_im_drt.max()*1000:.3f} mOhm*cm^2")

    return freq, Z_re_corrected, Z_im_drt, Rs
