# Dataset-for-SOC-Estimation-for-Large-Format-LFP-Batteries
Dataset for paper: SOC Estimation for Large-Format LFP Batteries Using EIS-DRT-Based Periodic Correction and Slope-Adaptive EKF Tracking

## Install
```bash
# Install dependencies
pip install -r requirements.txt

# Toolkit Installation
pip install -e .
```

## 包结构说明
```
battery_analysis/           # 核心包目录
├── __init__.py            # 包入口，导出所有公共接口
├── eis/                   # EIS分析模块
│   ├── reader.py          # EIS数据读取与预处理
│   ├── models.py          # EIS等效电路模型与拟合算法
│   └── plotting.py        # EIS可视化（Nyquist图、Bode图等）
├── drt/                   # DRT分析模块
│   ├── core.py            # DRT核心计算与优化拟合
│   └── plotting.py        # DRT可视化（2D/3D绘图）
├── cycle/                 # 循环数据分析模块
│   └── analyzer.py        # 电池循环数据解析与dQ/dV分析
└── utils/                 # 工具函数模块
    ├── io.py              # 文件读写与元数据解析
    └── fitting.py         # 拟合算法工具（双指数拟合等）

apps/                       # 应用脚本目录（可直接运行）
├── eis_analysis/          # EIS相关应用
│   ├── eis_batch_fit.py   # 批量EIS拟合工具
│   ├── eis_universal_fit.py  # 通用EIS拟合工具
│   ├── eis_visualization.py  # EIS可视化套件
│   └── soc_eis_3d_visualization.py  # SOC-EIS三维绘图
├── drt_analysis/          # DRT相关应用
│   ├── drt_analysis_cli.py  # DRT分析命令行工具
│   ├── drt_comparison_plot.py  # DRT对比绘图
│   ├── drt_3d_plot.py     # DRT三维可视化
│   └── eis_drt_integration.py  # EIS与DRT数据集成导出
└── cycle_analysis/        # 循环数据分析应用
    └── dqv_analysis_mit.py  # MIT数据集dQ/dV分析

20degC/                    # EIS data
...
```

### Common script examples
1. **EIS fit**：`apps/eis_analysis/eis_batch_fit.py`
  -Batch-process EIS data files
  -Automatically fit the equivalent circuit model
  -Output the fitted parameters and comparison plots

2. **DRT analysis - Fig.5**：`apps/drt_analysis/drt_analysis_cli.py`
  -Compute the distribution of relaxation times (DRT) from the EIS data
  -Output the gamma curve and peak information

3. **3D SOC-EIS plot as Fig.6**：`apps/eis_analysis/soc_eis_3d_visualization.py`

## Warning
1. All scripts use hard-coded absolute data paths by default. Please update these paths to match your local environment before running the code.
2. This repository currently provides basic EIS test results collected at multiple temperatures, along with the corresponding plotting scripts. Additional data will be added after the related paper is formally published.
