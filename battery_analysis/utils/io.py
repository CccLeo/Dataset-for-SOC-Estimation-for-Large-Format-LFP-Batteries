"""IO工具函数模块"""

import os
import re
import csv
from datetime import datetime


def parse_filename_metadata(filename):
    """从文件名解析元数据

    命名规则: #4-RPT1-75%-21h.txt

    Args:
        filename: 文件名

    Returns:
        包含电池ID、测试轮次、SOC、静置时间等信息的字典
    """
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


def save_parameters_to_csv(params, param_names, file_path, metadata=None):
    """保存参数到CSV文件

    Args:
        params: 参数数组
        param_names: 参数名称列表
        file_path: CSV文件路径
        metadata: 额外的元数据字典
    """
    if metadata is None:
        metadata = {}

    # 准备数据
    data = {'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    data.update(metadata)
    for name, val in zip(param_names, params):
        data[name] = val

    # 检查文件是否存在
    file_exists = os.path.exists(file_path)

    # 写入CSV
    with open(file_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(data)

    print(f"Parameters saved to: {file_path}")
