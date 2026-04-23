"""电池电化学数据分析工具包

提供EIS分析、DRT分析、电池循环数据分析、电压弛豫拟合等功能
"""

__version__ = "1.0.0"

# 导出EIS相关功能
from .eis.reader import read_eis_data
from .eis.models import (
    smart_initial_guess,
    get_initial_params,
    circuit_model2,
    calculate_lowfreq_weights,
    fit_model_optimized,
    calculate_metrics
)
from .eis.plotting import plot_results, plot_comparison_nyquist

# 导出DRT相关功能
from .drt.core import DRTAnalyzerFinal, read_and_preprocess_eis
from .drt.plotting import plot_detailed_results, plot_2x2_results

# 导出循环分析相关功能
from .cycle.analyzer import BatteryCycleAnalyzer, MyBatteryCycleAnalyzer

# 导出工具函数
from .utils.io import parse_filename_metadata, save_parameters_to_csv
from .utils.fitting import double_exp
