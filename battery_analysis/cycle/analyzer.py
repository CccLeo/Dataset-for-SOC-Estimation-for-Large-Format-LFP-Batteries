"""电池循环数据分析模块

提供电池充放电循环数据的读取、分析和绘图功能
支持 MIT 数据集和新威测试数据
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import os
import warnings

from ..utils.fitting import double_exp

warnings.filterwarnings('ignore')


class BatteryCycleAnalyzer:
    """
    用于分析 MIT电池循环数据的类，支持读取 CSV文件，提取特定循环的电压数据，
    并绘制时间-电压曲线。
    """

    def __init__(self, file_path, step=50, start_cycle=2, end_cycle=None):
        """
        初始化电池循环分析器

        参数:
            file_path (str): CSV数据文件的路径
            step (int): 每隔多少圈绘制一次曲线，默认为50
            start_cycle (int): 从第几圈开始绘制，默认为2
            end_cycle (int): 截止圈数，None代表全部圈数
        """
        self.file_path = file_path
        self.step = step
        self.start_cycle = start_cycle
        self.end_cycle = end_cycle
        self.data = None
        self.selected_cycles = None

        # 检查文件是否存在
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件 {file_path} 不存在")

        # 忽略警告
        warnings.filterwarnings('ignore')

    def load_data(self):
        """
        加载CSV文件数据
        """
        try:
            self.data = pd.read_csv(self.file_path)
            print(f"成功加载数据文件: {self.file_path}")
            return True
        except Exception as e:
            print(f"加载数据文件时出错: {e}")
            return False

    def select_cycles(self):
        """
        选择需要绘制的循环圈数
        """
        if self.data is None:
            print("请先加载数据")
            return False

        unique_cycles = self.data['Cycle_Index'].unique()
        # 从start_cycle开始，每隔step圈选择一次
        if self.end_cycle is None:
             self.selected_cycles = [cycle for cycle in unique_cycles
                                if cycle >= self.start_cycle and cycle % self.step == 0 and cycle <= len(self.data)-10]
        else:
            self.selected_cycles = [cycle for cycle in unique_cycles
                                    if cycle >= self.start_cycle and cycle % self.step == 0 and cycle <= self.end_cycle]

        if not self.selected_cycles:
            print(f"没有找到符合条件的循环圈数 (起始圈数: {self.start_cycle}, 步长: {self.step})")
            return False

        print(f"选择了以下循环圈数: {self.selected_cycles}")
        return True

    def plot_time_voltage(self, step_index=None, show_plot=True, save_path=None):
        """
        绘制时间-电压曲线

        参数:
            step_index (int): 步骤索引，可选想要的研究步骤例如10，写None则是全部循环数据
            show_plot (bool): 是否显示图表，默认为True
            save_path (str): 图表保存路径，如果为None则不保存
        """

        plt.figure(figsize=(12, 8))

        for cycle in self.selected_cycles:
            if step_index is None:
                filtered_data = self.data[(self.data['Cycle_Index'] == cycle)]
            else:
                filtered_data = self.data[(self.data['Cycle_Index'] == cycle) &
                                          (self.data['Step_Index'] == step_index)]

            start_time = filtered_data['Test_Time'].iloc[0]
            filtered_data['Adjusted_Time'] = filtered_data['Test_Time'] - start_time

            plt.plot(filtered_data['Adjusted_Time'], filtered_data['Voltage'],
                     label=f'Cycle {cycle}')

        plt.xlabel('Time')  # 设置x轴标签
        plt.ylabel('Voltage')  # 设置y轴标签
        plt.title('Time-Voltage Curve')  # 设置图表标题
        plt.legend(loc='best')  # 显示图例
        plt.grid(True)  # 显示网格

        if save_path:
            plt.savefig(save_path)
            print(f"图表已保存至: {save_path}")

        if show_plot:
            plt.show()
        else:
            plt.close()

        return plt.gcf()

    def plot_time_current(self, step_index=None, show_plot=True, save_path=None):
        """
        绘制时间-电流曲线

        参数:
            step_index (int): 步骤索引，可选想要的研究步骤例如10，写None则是全部循环数据
            show_plot (bool): 是否显示图表，默认为True
            save_path (str): 图表保存路径，如果为None则不保存
        """

        plt.figure(figsize=(12, 8))

        for cycle in self.selected_cycles:
            if step_index is None:
                filtered_data = self.data[(self.data['Cycle_Index'] == cycle)]
            else:
                filtered_data = self.data[(self.data['Cycle_Index'] == cycle) &
                                          (self.data['Step_Index'] == step_index)]

            start_time = filtered_data['Test_Time'].iloc[0]
            filtered_data['Adjusted_Time'] = filtered_data['Test_Time'] - start_time

            plt.plot(filtered_data['Adjusted_Time'], filtered_data['Current'],
                     label=f'Cycle {cycle}')

        plt.xlabel('Time')  # 设置x轴标签
        plt.ylabel('Current')  # 设置y轴标签
        plt.title('Time-Current Curve')  # 设置图表标题
        plt.legend(loc='best')  # 显示图例
        plt.grid(True)  # 显示网格

        if save_path:
            plt.savefig(save_path)
            print(f"图表已保存至: {save_path}")

        if show_plot:
            plt.show()
        else:
            plt.close()

        return plt.gcf()


class MyBatteryCycleAnalyzer:
    """
    用于分析 使用新威进行实验 的电池循环数据的类，支持读取 xlsx文件，提取特定循环的电压数据，
    并绘制时间-电压曲线。
    """

    def __init__(self, file_path, step=50, start_cycle=2, end_cycle=None):
        """
        初始化电池循环分析器

        参数:
            file_path (str): 数据文件的路径
            step (int): 每隔多少圈绘制一次曲线，默认为50
            start_cycle (int): 从第几圈开始绘制，默认为2
            end_cycle (int): 截止圈数，None代表全部圈数
        """
        self.file_path = file_path
        self.step = step
        self.start_cycle = start_cycle
        self.end_cycle = end_cycle
        self.data = None
        self.dataT = None
        self.selected_cycles = None

        # 检查文件是否存在
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件 {file_path} 不存在")

        # 忽略警告
        warnings.filterwarnings('ignore')

    def load_data(self):
        """
        加载文件数据，并合并主通道和辅助通道两个通道的数据
        """
        try:
            self.data = pd.read_excel(self.file_path, sheet_name='record') # 读取主通道数据
            self.dataT = pd.read_excel(self.file_path, sheet_name='auxTemp',header=1) # 读取温度通道数据
            self.data = pd.merge(self.data, self.dataT, on='数据序号', how='inner')
            self.data['总时间_s'] = pd.to_timedelta(self.data['总时间']).dt.total_seconds() # 总时间换算

            # print(f"成功加载数据文件: {self.file_path}")
            return True
        except Exception as e:
            print(f"加载数据文件时出错: {e}")
            return False

    def select_cycles(self):
        """
        选择需要绘制的循环圈数
        """
        if self.data is None:
            print("请先加载数据")
            return False

        unique_cycles = self.data['循环号'].unique()
        # 从start_cycle开始，每隔step圈选择一次
        if self.end_cycle is None:
             self.selected_cycles = [cycle for cycle in unique_cycles
                                if cycle >= self.start_cycle and cycle % self.step == 0 and cycle <= len(self.data)-10]
        else:
            self.selected_cycles = [cycle for cycle in unique_cycles
                                    if cycle >= self.start_cycle and cycle % self.step == 0 and cycle <= self.end_cycle]

        if not self.selected_cycles:
            print(f"没有找到符合条件的循环圈数 (起始圈数: {self.start_cycle}, 步长: {self.step})")
            return False

        # print(f"选择了以下循环圈数进行绘图: {self.selected_cycles}")
        return True

    def plot_time_voltage(self, step_index=None, show_plot=True, save_path=None, save_txt=False):
        """
        绘制时间-电压曲线

        参数:
            step_index (int): 工步索引，可选想要的研究步骤例如10，写None则是该循环下所有工步数据
            show_plot (bool): 是否显示图表，默认为True
            save_path (str): 图表保存路径，如果为None则不保存
        """
        fig, axs= plt.subplots(4, 1, figsize=(12, 18), sharex=True)
        ax1 = axs[0]
        ax2 = axs[1]
        ax3 = axs[2]
        ax4 = axs[3]

        lines_list = []

        for cycle in self.selected_cycles:
            if step_index is None:
                filtered_data = self.data[(self.data['循环号'] == cycle)]
            else:
                filtered_data = self.data[(self.data['循环号'] == cycle) &
                                          (self.data['工步号'] == step_index)]

            start_time = filtered_data['总时间_s'].iloc[0]
            filtered_data['Adjusted_Time'] = filtered_data['总时间_s'] - start_time

            color = plt.cm.viridis(cycle / len(self.selected_cycles))

            line1, = ax1.plot(filtered_data['Adjusted_Time'], filtered_data['电压(V)'],
                     linewidth=2.0, color=color, label=f'Cycle {cycle}')
            ax1.set_ylabel('Voltage (V)')

            ax1.grid(True, alpha=0.3)

            ax2.plot(filtered_data['Adjusted_Time'], filtered_data['T1'],
                     linewidth=2.0, color=color, label=f'Cycle {cycle}')
            ax2.set_ylabel('T1 (°C)')

            ax2.grid(True, alpha=0.3)

            ax3.plot(filtered_data['Adjusted_Time'], filtered_data['T2'],
                     linewidth=2.0, color=color, label=f'Cycle {cycle}')
            ax3.set_ylabel('T2 (°C)')

            ax3.grid(True, alpha=0.3)

            ax4.plot(filtered_data['Adjusted_Time'], filtered_data['电流(A)'],
                     linewidth=2.0, color=color, label=f'Cycle {cycle}')
            ax4.set_xlabel('Time (s)')
            ax4.set_ylabel('Current (A)')
            # ax4.legend(loc='best')
            ax4.grid(True, alpha=0.3)

            lines_list.append(line1)
            # 输出时间、电压数据到txt中，用于电化学模型对比
            if save_txt:
                 export_df = pd.DataFrame({
                'Time_s': filtered_data['Adjusted_Time'],
                'Voltage_V': filtered_data['电压(V)']
                 })
                 save_dir = r'E:\Datasets\314Ah\Model\export_txt'  # ← 改成自己的文件夹E:\Datasets\314Ah\Model
                 os.makedirs(save_dir, exist_ok=True)  # 如果不存在就创建

                 savepath = os.path.join(save_dir, f'cycle_{cycle}_time_voltage.txt')

                 export_df.to_csv(
                savepath,
                sep='\t',  # 用制表符分隔，方便用Excel或Origin打开
                index=False,  # 不保存行号
                float_format='%.6f'  # 控制小数位
                )

        filename = os.path.splitext(os.path.basename(self.file_path))[0]
        # fig.suptitle(f'{filename}')
        # plt.subplots_adjust(top=0.92)
        sm = cm.ScalarMappable(cmap='viridis',
                               norm=Normalize(vmin=min(self.selected_cycles),
                                              vmax=max(self.selected_cycles)))
        # plt.tight_layout()
        plt.tight_layout(rect=[0, 0, 0.88, 1])
        cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
        cbar=fig.colorbar(sm, cax=cbar_ax, ax=[ax1, ax2, ax3, ax4], shrink=0.95, label='Cycle Number')
        cbar.set_ticks([min(self.selected_cycles), max(self.selected_cycles)])
        if save_path:
            plt.savefig(save_path)
            print(f"图表已保存至: {save_path}")

        if show_plot:
            plt.show()
        else:
            plt.close()

        return plt.gcf()


    def plot_time_current(self, step_index=None, show_plot=True, save_path=None):
        """
        绘制时间-电流曲线

        参数:
            step_index (int): 步骤索引，可选想要的研究步骤例如10，写None则是全部循环数据
            show_plot (bool): 是否显示图表，默认为True
            save_path (str): 图表保存路径，如果为None则不保存
        """

        plt.figure(figsize=(12, 8))

        for cycle in self.selected_cycles:
            color = plt.cm.viridis(cycle / 25)
            if step_index is None:
                filtered_data = self.data[(self.data['循环号'] == cycle)]
            else:
                filtered_data = self.data[(self.data['循环号'] == cycle) &
                                          (self.data['工步号'] == step_index)]

            start_time = filtered_data['总时间_s'].iloc[0]
            filtered_data['Adjusted_Time'] = filtered_data['总时间_s'] - start_time
            filtered_data["dV"] = filtered_data["电压(V)"].diff()

            plt.plot(filtered_data['Adjusted_Time'], filtered_data["电压(V)"],color=color,
                     label=f'Cycle {cycle}')

        plt.xlabel('Time')  # 设置x轴标签
        plt.ylabel('Voltage')  # 设置y轴标签
        plt.title('Time-Voltage Curve')  # 设置图表标题
        plt.legend(loc='best')  # 显示图例
        plt.grid(True)  # 显示网格

        if save_path:
            plt.savefig(save_path)
            print(f"图表已保存至: {save_path}")

        if show_plot:
            plt.show()
        else:
            plt.close()

    def plot_feats(self, step_index=None):
        """
        绘制SOC-特征图，观察所选择的特征是否有效
        """
        plt.figure(figsize=(12, 8))

        for cycle in self.selected_cycles:
            color = plt.cm.Blues(cycle / 20)
            if step_index is None:
                filtered_data = self.data[(self.data['循环号'] == cycle)]
            else:
                filtered_data = self.data[(self.data['循环号'] == cycle) &
                                          (self.data['工步号'] == step_index)]

            start_time = filtered_data['总时间_s'].iloc[0]
            filtered_data['Adjusted_Time'] = filtered_data['总时间_s'] - start_time

            plt.scatter(cycle, filtered_data['电压(V)'].iloc[0],marker='o',label=f'1s')
            plt.scatter(cycle, filtered_data['电压(V)'].iloc[10],marker='*',label=f'2s')
            plt.scatter(cycle, filtered_data['电压(V)'].iloc[60],marker='+',label=f'60s')

            text_label = f"{filtered_data['电压(V)'].iloc[1]:.4f}V"
            plt.annotate(
                text_label,  # 要显示的文本
                xy=(cycle, filtered_data['电压(V)'].iloc[1]),  # 箭头/点指向的坐标 (你的*号位置)
                xytext=(5, 5),  # 文字偏移量 (x方向偏移5, y方向偏移5，单位是点)
                textcoords='offset points',  # 告诉系统 xytext 是偏移量，不是绝对坐标
                fontsize=10,  # 字体大小
                color='black'  # 字体颜色
            )
        plt.xlabel('Time')  # 设置x轴标签
        plt.ylabel('Voltage')  # 设置y轴标签
        plt.title('Time-Voltage Curve')  # 设置图表标题
        plt.legend(loc='best')  # 显示图例
        plt.grid(True)  # 显示网格
        plt.show()
        return plt.gcf()

    def calculate_feats(self, step_index=None):
        """
        计算特征与soc的关系度，观察所选择的特征是否有效
        """
        results = []

        for cycle in self.selected_cycles:
            if step_index is None:
                filtered_data = self.data[(self.data['循环号'] == cycle)]
            else:
                filtered_data = self.data[(self.data['循环号'] == cycle) &
                                          (self.data['工步号'] == step_index)]
            feats = {}
            feats["SOC"] = 1-cycle*0.05
            # 1->95% 2->90% ...20->0%
            second = 60
            feats["V1"] = filtered_data['电压(V)'].iloc[1]
            feats[f"V{second}"] = filtered_data['电压(V)'].iloc[second]
            results.append(feats)

        df_feats = pd.DataFrame(results)
        corr_matrix = df_feats.corr()
        print("特征相关性矩阵：")
        print(corr_matrix)
        v0_corr = corr_matrix.loc['SOC', 'V1']
        v1_corr = corr_matrix.loc['SOC', f"V{second}"]

        print(f"\nV0 与 SOC 的相关系数: {v0_corr:.4f}")
        print(f"V{second} 与 SOC 的相关系数: {v1_corr:.4f}")
        return df_feats

    def plot_all(self):
        """
        串行绘制时间-电流、电压、温度曲线在一张3Y轴的图中

        """
        fig, ax1 = plt.subplots(figsize=(12, 8))

        # ===== Y轴1：电压 =====
        ax1.plot(
            self.data['总时间_s'],
            self.data['电压(V)'],
            color='red',
            label='Voltage (V)'
        )
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Voltage (V)')
        ax1.grid(True)

        # ===== Y轴2：电流 =====
        ax2 = ax1.twinx()
        ax2.plot(
            self.data['总时间_s'],
            self.data['电流(A)'],
            label='Current (A)'
        )
        ax2.set_ylabel('Current (A)')

        # ===== Y轴3：温度 =====
        ax3 = ax1.twinx()

        # 关键：把第三个Y轴右移
        ax3.spines['right'].set_position(('outward', 60))

        ax3.plot(
            self.data['总时间_s'],
            self.data['T1'],
            label='T1 (°C)'
        )
        ax3.plot(
            self.data['总时间_s'],
            self.data['T2'],
            color='green',
            label='T2 (°C)'
        )
        ax3.set_ylabel('Temperature (°C)')

        # ===== 合并图例 =====
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines3, labels3 = ax3.get_legend_handles_labels()

        ax1.legend(
            lines1 + lines2 + lines3,
            labels1 + labels2 + labels3,
            loc='best'
        )

        plt.title('Battery Test Data (3 Y-Axes)')
        plt.tight_layout()
        plt.show()

    def plot_seprate(self):

        # 绘制时间-电流、电压、温度曲线在两张双Y轴的图中

        fig, (ax1, ax2) = plt.subplots(
            2, 1,
            figsize=(12, 9),
            sharex=True
        )

        # ================= 上图：电压 + 电流 =================
        ax1.plot(
            self.data['总时间_s'],
            self.data['电压(V)'],
            color='red',
            label='Voltage (V)'
        )
        ax1.set_ylabel('Voltage (V)')
        ax1.grid(True)

        ax1_i = ax1.twinx()
        ax1_i.plot(
            self.data['总时间_s'],
            self.data['电流(A)'],
            label='Current (A)'
        )
        ax1_i.set_ylabel('Current (A)')

        # 上图图例
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1_i.get_legend_handles_labels()
        ax1.legend(
            lines1 + lines2,
            labels1 + labels2,
            loc='best'
        )

        # ================= 下图：电流 + 温度 =================
        ax2.plot(
            self.data['总时间_s'],
            self.data['电流(A)'],
            label='Current (A)'
        )
        ax2.set_ylabel('Current (A)')
        ax2.grid(True)

        ax2_t = ax2.twinx()
        ax2_t.plot(
            self.data['总时间_s'],
            self.data['T1'],
            color='green',
            label='T1 (°C)'
        )
        ax2_t.plot(
            self.data['总时间_s'],
            self.data['T2'],
            color='red',
            label='T2 (°C)'
        )
        ax2_t.set_ylabel('Temperature (°C)')

        # 下图图例
        lines3, labels3 = ax2.get_legend_handles_labels()
        lines4, labels4 = ax2_t.get_legend_handles_labels()
        ax2.legend(
            lines3 + lines4,
            labels3 + labels4,
            loc='best'
        )

        # ================= 公共 X 轴 =================
        ax2.set_xlabel('Time (s)')

        plt.suptitle('Battery Test Data')
        plt.tight_layout()
        plt.show()

    def calculate_soc(self, nominal_capacity=325.0, initial_soc=1.0):
        # 确保数据按时间排序
        self.data = self.data.sort_values('总时间_s')

        # 计算时间差（秒）
        time_diff = np.diff(self.data['总时间_s'], prepend=0)

        # 计算安时变化量 (电流 * 时间 / 3600)
        ah_change = self.data['电流(A)'] * time_diff / 3600.0

        # 累积安时变化量
        cumulative_ah = np.cumsum(ah_change)

        # 计算SOC (初始SOC + 累积安时变化 / 电池容量)
        self.data['SOC'] = initial_soc + cumulative_ah / nominal_capacity

        # 限制SOC在0-1范围内
        self.data['SOC'] = np.clip(self.data['SOC'], 0, 1)
        # 绘制SOC
        plt.plot(self.data['总时间_s'], self.data['SOC'], 'b-', linewidth=2)
        plt.xlabel('Time(s)')
        plt.ylabel('SOC')
        plt.title('SOC')
        plt.grid(True)
        plt.ylim(0, 1.1)
        plt.show()
        return self.data
