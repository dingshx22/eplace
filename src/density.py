"""
密度图实现
用于静电模型中的密度计算和梯度计算
"""

import numpy as np
import logging
from scipy.ndimage import gaussian_filter
from typing import List, Dict, Tuple
from src.circuit import Cell
from src.utility import solve_poisson_fft, solve_poisson_dst, calculate_field


logger = logging.getLogger("ePlace.DensityMap")

class DensityMap:
    def __init__(self, origin_x: float, origin_y: float, width: float, height: float, bin_size: float = 10):
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.width = width
        self.height = height
        
        # 确保 bin_size 不为零
        self.bin_size = max(bin_size, 1.0)  # 设置最小网格大小为1.0
        
        # 使用 np.ceil 计算网格数量，确保完全覆盖区域
        self.nx = int(np.ceil(width / self.bin_size))
        self.ny = int(np.ceil(height / self.bin_size))
        
        # 重新计算实际的网格大小，以确保均匀覆盖
        self.bin_width = width / self.nx
        self.bin_height = height / self.ny
        self.bin_capacity = self.bin_width * self.bin_height
    
        # 初始化密度矩阵和电场
        self.density = np.zeros((self.nx, self.ny))        # 初始化密度矩阵
        self.potential = np.zeros((self.nx, self.ny))      # 电势场
        self.field_x = np.zeros((self.nx, self.ny))       # 电场强度
        self.field_y = np.zeros((self.nx, self.ny))
    
    def clear(self):
        self.density.fill(0.0)
        self.potential.fill(0.0)
        self.field_x.fill(0.0)
        self.field_y.fill(0.0)
    
    def add_cell(self, cell: Cell):
        start_x = int((cell.x - self.origin_x) / self.bin_width)
        start_y = int((cell.y - self.origin_y) / self.bin_height)
        end_x = int((cell.x + cell.width - self.origin_x) / self.bin_width) + 1
        end_y = int((cell.y + cell.height - self.origin_y) / self.bin_height) + 1
        
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
                bin_min_x = self.origin_x + x * self.bin_width
                bin_min_y = self.origin_y + y * self.bin_height
                bin_max_x = bin_min_x + self.bin_width
                bin_max_y = bin_min_y + self.bin_height
                
                overlap_min_x = max(bin_min_x, cell.x)
                overlap_min_y = max(bin_min_y, cell.y)
                overlap_max_x = min(bin_max_x, cell.x + cell.width)
                overlap_max_y = min(bin_max_y, cell.y + cell.height)
                
                # 计算重叠区域面积
                overlap_width = max(0, overlap_max_x - overlap_min_x)
                overlap_height = max(0, overlap_max_y - overlap_min_y)
                overlap_area = overlap_width * overlap_height
                
                # 更新密度（按面积比例分配）
                self.density[x, y] += overlap_area / self.bin_capacity
        
        # 更新电场
        self.update_field()
    
    def update_field(self):
        self.potential = solve_poisson_dst(self.density, (self.width, self.height), sigma=2)
        self.field_x, self.field_y = calculate_field(self.potential, (self.width, self.height))

    def get_density_at(self, x, y):
        bin_x = int((x - self.origin_x) / self.bin_width)        # 转换到网格坐标
        bin_y = int((y - self.origin_y) / self.bin_height)
    
        if bin_x < 0 or bin_x >= self.nx or bin_y < 0 or bin_y >= self.ny:        # 边界检查
            print(f"Error: DensityMap: get_density_at: bin_x = {bin_x}, bin_y = {bin_y}, nx = {self.nx}, ny = {self.ny}")
            return 0.0
        
        return self.density[bin_x, bin_y]
    
    def get_potential_at(self,x,y):
        bin_x = int((x - self.origin_x) / self.bin_width)        # 转换到网格坐标
        bin_y = int((y - self.origin_y) / self.bin_height)
    
        if bin_x < 0 or bin_x >= self.nx or bin_y < 0 or bin_y >= self.ny:        # 边界检查
            print(f"Error: DensityMap: get_potential_at: bin_x = {bin_x}, bin_y = {bin_y}, nx = {self.nx}, ny = {self.ny}")
            return 0.0
        
        return (self.field_x[bin_x, bin_y], self.field_y[bin_x, bin_y])        

    def get_max_density(self) -> float:
        return np.max(self.density)
    
    def get_average_density(self) -> float:
        return np.mean(self.density)
    
    def get_density_gradient(self, x: float, y: float) -> Tuple[float, float]: 
        bin_x = int((x - self.origin_x) / self.bin_width)
        bin_y = int((y - self.origin_y) / self.bin_height)
        
        if bin_x < 0 or bin_x >= self.nx or bin_y < 0 or bin_y >= self.ny:        # 边界检查
            print(f"Error: DensityMap: get_density_gradient: bin_x = {bin_x}, bin_y = {bin_y}, nx = {self.nx}, ny = {self.ny}")
            return (0.0, 0.0)
        return (self.field_x[bin_x, bin_y], self.field_y[bin_x, bin_y])        # 负梯度方向代表力的方向   

    def get_total_energy(self):
        # energy = 0.0
        # for _, cell in self.circuit.cells.items():
        # cx, cy = cell.get_center()
        # potential = self.get_potential_at(cx, cy)
        # energy += 0.5 * cell.get_area() * potential
        return 0.5 * np.sum(self.density * self.potential)