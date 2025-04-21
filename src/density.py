"""
密度图实现
用于静电模型中的密度计算和梯度计算
"""

import numpy as np
import logging
from scipy.ndimage import gaussian_filter
from typing import List, Dict, Tuple
from src.circuit import Cell
from src.utility import solve_poisson_fft, solve_poisson_dst


logger = logging.getLogger("ePlace.DensityMap")

class DensityMap:
    def __init__(self, origin_x: float, origin_y: float, width: float, height: float, grid_size: int = 10):
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.width = width
        self.height = height
        self.grid_size = grid_size
        
        # 初始化密度网格
        self.nx = grid_size
        self.ny = grid_size

        self.grid_width = width / grid_size
        self.grid_height = height / grid_size
    
        self.density = np.zeros((self.nx, self.ny))        # 初始化密度矩阵
        self.potential = np.zeros((self.nx, self.ny))        # 电势场

        self.field_x = np.zeros((self.nx, self.ny))        # 电场梯度
        self.field_y = np.zeros((self.nx, self.ny))
    
    def clear(self):
        self.density.fill(0.0)
        self.potential.fill(0.0)
        self.field_x.fill(0.0)
        self.field_y.fill(0.0)
    
    def add_cell(self, cell: Cell):
        start_x = int((cell.x - self.origin_x) / self.grid_width)
        start_y = int((cell.y - self.origin_y) / self.grid_height)
        end_x = int((cell.x + cell.width - self.origin_x) / self.grid_width) + 1
        end_y = int((cell.y + cell.height - self.origin_y) / self.grid_height) + 1
        
        # 裁剪到有效范围，确保网格的范围不会超出密度图的边界
        start_x = max(0, min(self.nx - 1, start_x))
        start_y = max(0, min(self.ny - 1, start_y))
        end_x = max(0, min(self.nx, end_x))
        end_y = max(0, min(self.ny, end_y))
        
        # cell_area = cell.get_area()
        
        # 将单元密度分配到覆盖的网格
        for x in range(start_x, end_x):
            for y in range(start_y, end_y):
                # 计算单元与网格的重叠区域
                grid_min_x = self.origin_x + x * self.grid_width
                grid_min_y = self.origin_y + y * self.grid_height
                grid_max_x = grid_min_x + self.grid_width
                grid_max_y = grid_min_y + self.grid_height
                
                overlap_min_x = max(grid_min_x, cell.x)
                overlap_min_y = max(grid_min_y, cell.y)
                overlap_max_x = min(grid_max_x, cell.x + cell.width)
                overlap_max_y = min(grid_max_y, cell.y + cell.height)
                
                # 计算重叠区域面积
                overlap_width = max(0, overlap_max_x - overlap_min_x)
                overlap_height = max(0, overlap_max_y - overlap_min_y)
                overlap_area = overlap_width * overlap_height
                
                # 更新密度（按面积比例分配）
                self.density[x, y] += overlap_area / (self.grid_width * self.grid_height)
        
        # 更新电场
        self.update_field()
    
    def update_field(self):
        """更新静电场"""
        # # 对密度进行高斯平滑，模拟静电场扩散
        # smoothed_density = gaussian_filter(self.density, sigma=self.sigma)
        # # 计算静电势场（使用泊松方程 ∇²φ = -ρ）简化处理：直接使用平滑后的密度作为势场
        # self.potential = smoothed_density
        # # 计算电场梯度 (E = -∇φ)
        # self.field_x = np.zeros_like(self.potential)
        # self.field_y = np.zeros_like(self.potential)        
        # self.field_x[:-1, :] = -(self.potential[1:, :] - self.potential[:-1, :]) / self.grid_width   # x方向梯度
        # self.field_y[:, :-1] = -(self.potential[:, 1:] - self.potential[:, :-1]) / self.grid_height  # y方向梯度
        # Solve Poisson equation using DST with DC removal

        self.potential = solve_poisson_dst(self.density, (self.width, self.height), sigma=2)
        
        # Initialize field arrays
        self.field_x = np.zeros_like(self.potential)
        self.field_y = np.zeros_like(self.potential)
        
        # Central difference for x-direction gradient (second-order accurate)
        self.field_y[1:-1, :] = -(self.potential[2:, :] - self.potential[:-2, :]) / (2 * self.grid_height)
        # One-sided difference for boundaries
        self.field_y[0, :] = -(self.potential[1, :] - self.potential[0, :]) / self.grid_height
        self.field_y[-1, :] = -(self.potential[-1, :] - self.potential[-2, :]) / self.grid_height
        
        # Central difference for y-direction gradient
        self.field_x[:, 1:-1] = -(self.potential[:, 2:] - self.potential[:, :-2]) / (2 * self.grid_width)
        # One-sided difference for boundaries
        self.field_x[:, 0] = -(self.potential[:, 1] - self.potential[:, 0]) / self.grid_width
        self.field_x[:, -1] = -(self.potential[:, -1] - self.potential[:, -2]) / self.grid_width

    def get_density_at(self, x: float, y: float) -> float:
        """获取指定位置的密度值"""
        # 转换到网格坐标
        grid_x = int((x - self.origin_x) / self.grid_width)
        grid_y = int((y - self.origin_y) / self.grid_height)
    
        if grid_x < 0 or grid_x >= self.nx or grid_y < 0 or grid_y >= self.ny:        # 边界检查
            print(f"Error: DensityMap: get_density_at: grid_x = {grid_x}, grid_y = {grid_y}, nx = {self.nx}, ny = {self.ny}")
            return 0.0
        
        return self.density[grid_x, grid_y]
    
    def get_max_density(self) -> float:
        return np.max(self.density)
    
    def get_average_density(self) -> float:
        return np.mean(self.density)
    
    def get_density_gradient(self, x: float, y: float) -> Tuple[float, float]: 
        grid_x = int((x - self.origin_x) / self.grid_width)
        grid_y = int((y - self.origin_y) / self.grid_height)
        
        if grid_x < 0 or grid_x >= self.nx or grid_y < 0 or grid_y >= self.ny:        # 边界检查
            print(f"Error: DensityMap: get_density_gradient: grid_x = {grid_x}, grid_y = {grid_y}, nx = {self.nx}, ny = {self.ny}")
            return (0.0, 0.0)
        return (self.field_x[grid_x, grid_y], self.field_y[grid_x, grid_y])        # 负梯度方向代表力的方向