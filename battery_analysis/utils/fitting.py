"""拟合工具函数模块

包含各种数学拟合函数，用于电池数据分析
"""

import numpy as np


def double_exp(t, V_inf, A1, tau1, A2, tau2):
    """双指数衰减函数

    用于拟合电池电压弛豫过程

    Args:
        t: 时间数组
        V_inf: 稳态电压
        A1: 第一个指数项的幅值
        tau1: 第一个时间常数
        A2: 第二个指数项的幅值
        tau2: 第二个时间常数

    Returns:
        拟合值数组
    """
    return V_inf + A1 * np.exp(-t / tau1) + A2 * np.exp(-t / tau2)
