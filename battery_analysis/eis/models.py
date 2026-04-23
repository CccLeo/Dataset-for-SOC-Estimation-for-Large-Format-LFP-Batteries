"""EIS circuit model and fitting module"""

import numpy as np
from scipy.optimize import least_squares


def smart_initial_guess(freq, Z_re, Z_im):
    """Smart initial parameter estimation

    Args:
        freq: frequency array
        Z_re: real part array
        Z_im: imaginary part array

    Returns:
        dictionary containing initial parameter estimates
    """
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
    """Generate initial parameters for Model 2 based on initial estimates

    Args:
        init_dict: smart initial estimation dictionary

    Returns:
        list of initial parameters
    """
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
    """Model 2: Rs + (R1||CPE1) + (R2||CPE2) + Bounded Warburg

    Args:
        params: model parameters [Rs, R1, Q1, a1, R2, Q2, a2, sigma, A, B]
        omega: angular frequency array

    Returns:
        complex impedance array
    """
    Rs, R1, Q1, a1, R2, Q2, a2, sigma, A, B = params
    Z1 = 1 / (1/R1 + (Q1 * (1j*omega)**a1))
    Z2 = 1 / (1/R2 + (Q2 * (1j*omega)**a2))
    s = np.sqrt(1j * omega)
    Zw = sigma * np.tanh(A * s) / (s * (1 + B * s))
    return Rs + Z1 + Z2 + Zw


def calculate_lowfreq_weights(freq, Z_re, Z_im, lowfreq_threshold=0.5):
    """Calculate weights emphasizing low frequencies

    Args:
        freq: frequency array
        Z_re: real part array
        Z_im: imaginary part array
        lowfreq_threshold: low frequency threshold

    Returns:
        weight array
    """
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
    """Fit Model 2

    Args:
        params_init: initial parameters
        freq: frequency array
        Z_re: real part array
        Z_im: imaginary part array

    Returns:
        fitting result object
    """
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
    """Calculate fitting metrics

    Args:
        Z_fit: fitted impedance array
        Z_re: experimental real part array
        Z_im: experimental imaginary part array
        freq: frequency array

    Returns:
        dictionary containing various metrics
    """
    # Low frequency threshold
    LOW_FREQ_THRESHOLD = 1.0

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
