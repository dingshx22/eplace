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
        
        # 创建图形，调整大小和布局
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_axes([0.05, 0.05, 0.8, 0.9]) #colorbar预留空间
        plt.ion()  # 打开交互模式

        logger.info("可视化器初始化完成")

    def clear_plot(self):
        """清除当前图形内容"""
        self.ax.clear()
        
        # 清除 colorbar
        if hasattr(self, 'colorbar') and self.colorbar is not None:
            try:
                self.colorbar.ax.remove()
                self.colorbar = None
            except Exception as e:
                logger.warning(f"清除 colorbar 时出错: {e}")
        
        # 清除存储的图形对象
        self.cell_patches.clear()
        self.cell_texts.clear()
        
        # 重置坐标轴设置和位置
        self.ax.set_position([0.05, 0.05, 0.8, 0.9])  # 重新设置主图位置
        # self.ax.set_xlabel('X 坐标')
        # self.ax.set_ylabel('Y 坐标')
        
        # 重置图形布局
        self.fig.canvas.draw()

    def visualize_potential(self, density_map: DensityMap = None, output_file = None,show =True):
        """可视化电势图"""
        # 清除之前的内容
        self.clear_plot()
        
        # 获取电势数据
        potential = density_map.potential
        
        # 创建网格
        x = np.linspace(density_map.origin_x, density_map.origin_x + density_map.width, density_map.nx)
        y = np.linspace(density_map.origin_y, density_map.origin_y + density_map.height, density_map.ny)
        X, Y = np.meshgrid(x, y)
        
        
        # 绘制电势图
        # # contour = self.ax.contourf(X, Y, potential.T, levels=20, cmap='viridis')  
        # self.colorbar = plt.colorbar(contour, ax=self.ax, label='电势', pad=0.05)

        img = self.ax.imshow(
        potential.T,                      # 注意转置，使图的方向正确
        extent=[
            density_map.origin_x,
            density_map.origin_x + density_map.width,
            density_map.origin_y,
            density_map.origin_y + density_map.height
        ],
        origin='lower',                   # 原点在左下角
        cmap='viridis',                   # 使用相同颜色映射
        aspect='equal'                    # 坐标比例一致
    )

    # 添加颜色条
        self.colorbar = plt.colorbar(img, ax=self.ax, label='电势', pad=0.05)        

        # 设置坐标轴
        self.ax.set_xlim(density_map.origin_x, density_map.origin_x + density_map.width)
        self.ax.set_ylim(density_map.origin_y, density_map.origin_y + density_map.height)
        self.ax.set_aspect('equal')
        # self.ax.set_xlabel('X 坐标')
        # self.ax.set_ylabel('Y 坐标')
        self.ax.set_title('电势分布图', pad=20)
        
        # 保存或显示图形
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            # logger.info(f"电势图保存到: {output_file}")
        
        # 更新显示
        if show:
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

    def visualize_density(self, density_map: DensityMap=None, output_file = None,show=True):
        """可视化密度图"""
        self.clear_plot()
        density = density_map.density
        
        # 创建网格
        x = np.linspace(density_map.origin_x, density_map.origin_x + density_map.width, density_map.nx)
        y = np.linspace(density_map.origin_y, density_map.origin_y + density_map.height, density_map.ny)
        X, Y = np.meshgrid(x, y)
        
        # 绘制密度图
        # contour = self.ax.contourf(X, Y, density.T, levels=20, cmap='hot')
        # self.colorbar = plt.colorbar(contour, ax=self.ax, label='密度', pad=0.05)
    # 用 imshow 绘制热力图
        img = self.ax.imshow(
        density.T,                      # 转置矩阵，使方向正确
        extent=[
            density_map.origin_x,
            density_map.origin_x + density_map.width,
            density_map.origin_y,
            density_map.origin_y + density_map.height
        ],
        origin='lower',                 # 原点设在左下角（更直观）
        cmap='hot',                     # 热力图配色方案
        aspect='equal'                  # 保持坐标比例一致
    )
    # 添加颜色条
        self.colorbar = plt.colorbar(img, ax=self.ax, label='密度', pad=0.05)
        
        # 设置坐标轴
        self.ax.set_xlim(density_map.origin_x, density_map.origin_x + density_map.width)
        self.ax.set_ylim(density_map.origin_y, density_map.origin_y + density_map.height)
        self.ax.set_aspect('equal')
        # self.ax.set_xlabel('X 坐标')
        # self.ax.set_ylabel('Y 坐标')
        self.ax.set_title('单元密度分布图', pad=20)
        
        # 保存或显示图形
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            # logger.info(f"密度图保存到: {output_file}")
        
        # 更新显示
        if show:
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

    def draw_movement_arrows(self, gradients: Dict[str, Tuple[float, float]]):
        """
        Args:
            gradients: 字典，键为单元名称，值为(grad_x, grad_y)梯度元组
        """
        if not gradients:
            return
            
        for cell_name, (grad_x, grad_y) in gradients.items():
            if cell_name not in self.circuit.cells:
                continue
                
            cell = self.circuit.cells[cell_name]
            # 获取单元中心点
            center_x, center_y = cell.get_center()
            
            # 计算箭头长度（根据梯度大小标准化）
            magnitude = np.sqrt(grad_x**2 + grad_y**2)
            if magnitude < 1e-6:  # 避免除以零
                continue
                
            # 标准化箭头长度
            arrow_length = min(20, magnitude * 0.1)  # 限制最大长度
            dx = -grad_x * arrow_length / magnitude  # 注意这里是负的梯度方向
            dy = -grad_y * arrow_length / magnitude
            


            # 绘制箭头
            arrow = Arrow(center_x, center_y, dx, dy,
                        width=0.5, color='black', alpha=0.8)
            self.ax.add_patch(arrow)

    def visualize_placement(self, density_map: DensityMap = None, show_field = False, show = True, output_file = None, gradients: Dict[str, Tuple[float, float]] = None):
        """可视化布局结果
        Args:
            density_map: 密度图对象
            show_field: 是否显示电场
            show: 是否显示图形
            output_file: 输出文件路径
            gradients: 单元移动方向的梯度字典
        """       
        # 清除之前的内容
        self.ax.clear()
        self.cell_patches.clear()
        self.cell_texts.clear()
        
        # 清除 colorbar
        if hasattr(self, 'colorbar') and self.colorbar is not None:
            try:
                self.colorbar.ax.remove()
                self.colorbar = None
            except Exception as e:
                logger.warning(f"清除 colorbar 时出错: {e}")
        
        # 重置坐标轴设置和位置
        self.ax.set_position([0.1, 0.1, 0.85, 0.8])  # 布局图不需要为colorbar预留空间
        
        # 绘制芯片边界
        min_x, min_y, max_x, max_y = self.circuit.die_area
        self.ax.add_patch(Rectangle((min_x, min_y), max_x - min_x, max_y - min_y, fill=False, edgecolor='black',linewidth=2))
        # 绘制单元
        self.draw_cells()
        
        # 如果提供了密度图，显示电场
        if density_map is not None and show_field:
            self.draw_field(density_map)
            
        # 如果提供了梯度信息，绘制移动方向
        if gradients is not None:
            self.draw_movement_arrows(gradients)
        
        # 设置坐标轴
        self.ax.set_xlim(min_x - 1, max_x + 1)
        self.ax.set_ylim(min_y - 1, max_y + 1)
        self.ax.set_aspect('equal')
        # self.ax.set_xlabel('X 坐标')
        # self.ax.set_ylabel('Y 坐标')
        
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
            # logger.info(f"布局图保存到: {output_file}")
        
        # 更新显示
        if show:
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