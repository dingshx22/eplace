"""
布局结果可视化
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Arrow
import numpy as np
import logging
from typing import List, Dict, Tuple
from src.circuit import Circuit, Cell
from src.density import DensityMap

logger = logging.getLogger("ePlace.Visualizer")

class PlacementVisualizer:
    """布局结果可视化器"""
    def __init__(self, circuit: Circuit):
        self.circuit = circuit
        # 设置中文字体支持
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
        
        # 存储图形对象
        self.fig = None
        self.ax = None
        self.cell_patches = {}
        self.cell_texts = {}
        self.colorbar = None  # 添加 colorbar 引用
        
        # 创建图形
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        plt.ion()  # 打开交互模式
        logger.info("可视化器初始化完成")

    def clear_plot(self):
        """清除当前图形"""        # 完全重置图形
        plt.close(self.fig)
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        plt.ion()  # 重新打开交互模式
        
        # 重置所有属性
        self.cell_patches.clear()
        self.cell_texts.clear()
        self.colorbar = None

    def visualize_potential(self, density_map: DensityMap, output_file: str = None):
        """可视化电势图"""
        self.clear_plot()        # 清除之前的内容
        
        # 获取电势数据
        potential = density_map.potential
        
        # 创建网格
        x = np.linspace(density_map.origin_x, density_map.origin_x + density_map.width, density_map.nx)
        y = np.linspace(density_map.origin_y, density_map.origin_y + density_map.height, density_map.ny)
        X, Y = np.meshgrid(x, y)
        
        # 绘制电势图
        contour = self.ax.contourf(X, Y, potential.T, levels=20, cmap='viridis')
        self.colorbar = plt.colorbar(contour, ax=self.ax, label='电势', pad=0.1)
        
        # 设置坐标轴
        self.ax.set_xlim(density_map.origin_x, density_map.origin_x + density_map.width)
        self.ax.set_ylim(density_map.origin_y, density_map.origin_y + density_map.height)
        self.ax.set_aspect('equal')
        self.ax.set_xlabel('X 坐标')
        self.ax.set_ylabel('Y 坐标')
        self.ax.set_title('电势分布图')
        
        # 调整布局
        self.fig.tight_layout()
        
        # 保存或显示图形
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"电势图保存到: {output_file}")
        
        # 更新显示
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def visualize_density(self, density_map: DensityMap, output_file: str = None):
        """可视化密度图"""
        self.clear_plot()
        density = density_map.density
        
        # 创建网格
        x = np.linspace(density_map.origin_x, density_map.origin_x + density_map.width, density_map.nx)
        y = np.linspace(density_map.origin_y, density_map.origin_y + density_map.height, density_map.ny)
        X, Y = np.meshgrid(x, y)
        
        # 绘制密度图
        contour = self.ax.contourf(X, Y, density.T, levels=20, cmap='hot')
        self.colorbar = plt.colorbar(contour, ax=self.ax, label='密度', pad=0.1)
        
        # 设置坐标轴
        self.ax.set_xlim(density_map.origin_x, density_map.origin_x + density_map.width)
        self.ax.set_ylim(density_map.origin_y, density_map.origin_y + density_map.height)
        self.ax.set_aspect('equal')
        self.ax.set_xlabel('X 坐标')
        self.ax.set_ylabel('Y 坐标')
        self.ax.set_title('单元密度分布图')
        
        # 调整布局
        self.fig.tight_layout()
        
        # 保存或显示图形
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"密度图保存到: {output_file}")
        
        # 更新显示
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def visualize_placement(self, density_map: DensityMap = None, show_field: bool = False, output_file: str = None):
        """可视化布局结果"""       
        # 清除之前的内容
        self.ax.clear()
        self.cell_patches.clear()
        self.cell_texts.clear()
        
        # 绘制芯片边界
        min_x, min_y, max_x, max_y = self.circuit.die_area
        self.ax.add_patch(Rectangle((min_x, min_y), max_x - min_x, max_y - min_y, fill=False, edgecolor='black',linewidth=2))
        # 绘制单元
        self.draw_cells()
        
        # 如果提供了密度图，显示电场
        if density_map is not None and show_field:
            self.draw_field(density_map)
        
        # 设置坐标轴
        self.ax.set_xlim(min_x - 1, max_x + 1)
        self.ax.set_ylim(min_y - 1, max_y + 1)
        self.ax.set_aspect('equal')
        self.ax.set_xlabel('X 坐标')
        self.ax.set_ylabel('Y 坐标')
        
        # 根据文件名设置不同的标题
        title = '电路布局结果'
        self.ax.set_title(title)
        
        # 添加图例
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', alpha=0.7, label='固定单元'),
            Patch(facecolor='blue', alpha=0.7, label='宏单元'),
            Patch(facecolor='green', alpha=0.7, label='标准单元')
        ]
        if show_field:
            legend_elements.append(Arrow(0, 0, 1, 1, color='red', label='电场方向'))
        self.ax.legend(handles=legend_elements, loc='upper left')
        
        # 保存或显示图形
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"布局图保存到: {output_file}")
        
        # 更新显示
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
    def draw_cells(self):
        """绘制所有单元"""
        min_size = 8
        for cell_name, cell in self.circuit.cells.items():
            # 根据单元类型选择颜色
            if cell.is_fixed:
                color = 'red'
            elif cell.is_macro:
                color = 'blue'
            else:
                color = 'green'            
            # 绘制单元
            patch = Rectangle((cell.x, cell.y), cell.width, cell.height,
                            fill=True, alpha=0.7, facecolor=color,
                            edgecolor='black', linewidth=0.5)
            self.ax.add_patch(patch)
            self.cell_patches[cell_name] = patch
            
            # 为大型单元添加标签
            if cell.is_macro or cell.is_fixed or cell.width > min_size or cell.height > min_size:
                text = self.ax.text(cell.x + cell.width/2, cell.y + cell.height/2,
                                  cell_name, ha='center', va='center',
                                  fontsize=8, alpha=0.7)
                self.cell_texts[cell_name] = text
                
    def draw_field(self, density_map: DensityMap):
        """绘制电场方向"""
        # 清除之前的所有箭头
        self.ax.clear()  # 清除整个图形
        self.draw_cells()  # 重新绘制单元
        
        # 计算箭头的缩放因子
        scale = min(density_map.grid_width, density_map.grid_height) * 0.5
        
        # 在每个网格中心绘制电场方向
        for i in range(density_map.nx):
            for j in range(density_map.ny):
                # 计算网格中心坐标
                x = density_map.origin_x + (i + 0.5) * density_map.grid_width
                y = density_map.origin_y + (j + 0.5) * density_map.grid_height
                
                # 获取该点的电场
                fx, fy = density_map.field_x[i, j], density_map.field_y[i, j]
                # 如果电场强度足够大，才绘制箭头
                field_strength = np.sqrt(fx*fx + fy*fy)
                if field_strength > 0:  # 设置一个阈值
                    # 归一化并缩放
                    fx = fx / field_strength * scale
                    fy = fy / field_strength * scale                 
                    # 绘制箭头
                    self.ax.arrow(x, y, fx, fy,head_width=scale*0.2,head_length=scale*0.3,fc='red', ec='red',alpha=0.5)