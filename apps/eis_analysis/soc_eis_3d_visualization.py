"""
SOC-EIS 3D Plotting Module
Generates 3D plots from SOC-EIS.xlsx data, with x=SOC, y=Temperature, z=Variable value
Academic paper plotting style referenced from DRT-3D.py
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Font and minus sign display settings - consistent with DRT-3D.py
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.default'] = 'regular'

# Academic paper style settings - consistent with DRT-3D.py
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['xtick.major.width'] = 1.0
plt.rcParams['ytick.major.width'] = 1.0


def plot_soc_eis_3d(excel_path, save_dir='./output/', view_elev=19, view_azim=58):
    """
    Generate 3D surface plots for the three variables Z', Z'', and gamma
    Z'' uses a separately specified azimuth angle

    Args:
        excel_path: Excel file path
        save_dir: Image save directory
        view_elev: Elevation angle
        view_azim: Azimuth angle (used for Z' and gamma)

    Returns:
        figs: List of figure objects
    """
    # 读取数据
    df = pd.read_excel(excel_path)

    # 清理列名中的空格
    df.columns = df.columns.str.strip()
    print(f"Data shape: {df.shape}")
    print(f"Column names: {df.columns.tolist()}")

    # Define temperatures and variables, Z'' uses separate azimuth angle
    temperatures = [30, 25, 20, 15]
    variables = [
        {'name': "Z'", 'col_template': "Z'-{temp}", 'z_label': "$Z'$ ($\\Omega$)", 'azim': view_azim},
        {'name': "Z''", 'col_template': "Z''-{temp}", 'z_label': "$Z''$ ($\\Omega$)", 'azim': -120},
        {'name': "gama", 'col_template': "gama-{temp}", 'z_label': "$\\gamma$", 'azim': view_azim},
    ]

    # SOC数据
    soc_values = df['SOC'].values
    print(f"\nSOC range: [{soc_values.min():.2f}, {soc_values.max():.2f}]")
    print(f"Temperatures: {temperatures}")

    # Create save directory
    import os
    os.makedirs(save_dir, exist_ok=True)

    figs = []

    # Create a 3D plot for each variable
    for var in variables:
        print(f"\nPlotting: {var['name']}, azimuth={var['azim']}")

        # Prepare grid data
        X, Y = np.meshgrid(soc_values, temperatures)
        Z = np.zeros_like(X)

        # Populate Z data
        for i, temp in enumerate(temperatures):
            col_name = var['col_template'].format(temp=temp)
            Z[i, :] = df[col_name].values

        # 创建3D图形
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Plot surface - uses viridis color scheme (consistent with DRT-3D.py)
        surf = ax.plot_surface(X, Y, Z,
                              cmap='viridis',
                              alpha=0.85,
                              edgecolor='none',
                              antialiased=True,
                              zorder=5)

        # Plot contour lines at each temperature
        for i, temp in enumerate(temperatures):
            ax.plot(soc_values,
                   np.ones_like(soc_values) * temp,
                   Z[i, :],
                   linewidth=2,
                   color=plt.cm.viridis(i / len(temperatures)),
                   zorder=10)

        # Set axis labels - increase Z-axis labelpad to ensure full display
        ax.set_xlabel('SOC', fontsize=14, fontweight='bold', labelpad=10)
        ax.set_ylabel('Temperature (°C)', fontsize=14, fontweight='bold', labelpad=10)
        ax.set_zlabel(var['z_label'], fontsize=14, fontweight='bold', labelpad=25)

        # Set Y-axis ticks and range
        ax.set_yticks(temperatures)
        ax.set_yticklabels([f'{temp}' for temp in temperatures], fontsize=11)
        temp_range = max(temperatures) - min(temperatures)
        ax.set_ylim(min(temperatures) - temp_range * 0.1, max(temperatures) + temp_range * 0.1)

        # Set X-axis ticks
        ax.set_xticks(np.arange(0.2, 1.0, 0.1))
        ax.set_xticklabels([f'{tick:.1f}' for tick in np.arange(0.2, 1.0, 0.1)], fontsize=11)
        ax.set_xlim(soc_values.min() - 0.02, soc_values.max() + 0.02)

        # Set Z-axis range
        z_min, z_max = Z.min(), Z.max()
        z_range = z_max - z_min
        ax.set_zlim(z_min - z_range * 0.1, z_max + z_range * 0.1)

        # Use scientific notation for Z-axis ticks to avoid long number overlap
        ax.ticklabel_format(axis='z', style='sci', scilimits=(0, 0))

        # Set view angle - each variable uses its own azimuth angle
        ax.view_init(elev=view_elev, azim=var['azim'])

        # Set 3D box aspect ratio
        ax.set_box_aspect((1, 0.8, 0.6))

        # Add XY plane base
        x_plane = np.linspace(soc_values.min(), soc_values.max(), 20)
        y_plane = np.linspace(min(temperatures) - temp_range * 0.1,
                             max(temperatures) + temp_range * 0.1, 2)
        X_plane, Y_plane = np.meshgrid(x_plane, y_plane)
        Z_plane = np.zeros_like(X_plane) + z_min - z_range * 0.05

        ax.plot_surface(X_plane, Y_plane, Z_plane,
                       alpha=0.05,
                       color='blue')

        # Set empty title
        ax.set_title('', pad=0)

        # Save image
        save_path = os.path.join(save_dir, f'SOC-EIS-3D-{var["name"]}.png')
        fig.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
        print(f"  Saved to: {save_path}")

        figs.append(fig)

    print("\n全部绘制完成!")
    return figs


if __name__ == "__main__":
    # 主程序
    excel_path = r"E:\PycharmProjects\PythonProject1\SOC-EIS.xlsx"
    save_dir = r"E:\PycharmProjects\PythonProject1\output"

    # Z'和gama: elevation=19, azimuth=58
    # Z'': elevation=19, azimuth=-120
    figs = plot_soc_eis_3d(excel_path, save_dir, view_elev=19, view_azim=58)

    plt.show()
