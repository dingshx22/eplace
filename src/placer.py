"""
基于静电力学的布局器实现
"""
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
from src.circuit import Circuit, Cell
from src.density import DensityMap
from src.visualizer import PlacementVisualizer
from scipy.ndimage import gaussian_filter
import time
import os
import csv

logger = logging.getLogger("ePlace.Placer")

TIMEs=1
HPWL_ref=30
COF_max=1.05
COF_min=0.95



class ElectrostaticPlacer:
    def __init__(self, circuit: Circuit, alpha = 1.0, max_iterations = 100, bin_size=5):
        self.circuit = circuit
        self.alpha = alpha        # 静电力学模型参数
        self._beta = 1          # 线长权重
        self._lambda = 0        # 密度权重
        self.max_iterations = max_iterations 
        self.bin_size = bin_size
            
        # 初始化密度图
        self.density_map = DensityMap(self.circuit.die_area[0], self.circuit.die_area[1], \
                                    self.circuit.die_area[2], self.circuit.die_area[3], bin_size=self.bin_size)  
            
        self.learning_rate = 0.1  
        self.visualizer = PlacementVisualizer(self.circuit)  
        logger.info(f"初始化布局器: alpha={self.alpha}, beta={self._beta}, gamma={self._lambda},max_iterations={self.max_iterations}")

    def initialize(self):        
        min_x, min_y, max_x, max_y = self.circuit.die_area      
        for _, cell in self.circuit.cells.items():    # 初始随机布局（对于未固定的单元）
            if not cell.is_fixed:
                if cell.is_macro:
                    # 大型宏单元放在边缘
                    edge_margin = 50
                    max_x = max_x - cell.width - edge_margin
                    max_y = max_y - cell.height - edge_margin
                    min_x_loc = min_x + edge_margin
                    min_y_loc = min_y + edge_margin
                else:
                    min_x_loc = min_x# 标准单元随机分布
                    min_y_loc = min_y
                    
                cell.x = np.random.uniform(min_x_loc, max_x - cell.width)
                cell.y = np.random.uniform(min_y_loc, max_y - cell.height)
        
        # 初始化密度图
        self.update_density_map()
                
    def update_density_map(self):
        self.density_map.clear()
        for _, cell in self.circuit.cells.items():
            self.density_map.add_cell(cell)

    def update_lambda(self,delta_hpwl,grad_wirelength,grad_potential):
        self._lambda = grad_wirelength/grad_potential
        p=delta_hpwl/HPWL_ref
        if p<0:
            cof=COF_max
        else:
            cof=max(COF_min,pow(COF_max,1-p))
        self._lambda=self._lambda*cof

    def calculate_wirelength_gradient(self, cell: Cell) -> Tuple[float, float]:
        """
        使用对数求和模型 (LSE - Log-Sum-Exp) 计算线长梯度
        """
        grad_x, grad_y = 0.0, 0.0
        
        # 获取与该单元相连的所有网络
        related_nets = []
        for _, net in self.circuit.nets.items():
            for pin in net.pins:
                if pin.parent_cell.name == cell.name:
                    related_nets.append(net)
                    break
        
        # 对每个相关网络计算梯度
        for net in related_nets:
            if len(net.pins) <= 1:
                continue
                
            positions = [pin.absolute_position for pin in net.pins]
            xs, ys = zip(*positions)
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            
            # LSE参数
            alpha = 0.01 * max(max_x - min_x, max_y - min_y)
            if alpha < 1e-6:  # 避免除以零
                alpha = 1e-6
                
            # 计算x方向梯度
            for pin in net.pins:
                if pin.parent_cell.name == cell.name:
                    x_pos = pin.absolute_position[0]
                    grad_x += np.tanh((x_pos - min_x) / alpha) - np.tanh((max_x - x_pos) / alpha)
            
            # 计算y方向梯度
            for pin in net.pins:
                if pin.parent_cell.name == cell.name:
                    y_pos = pin.absolute_position[1]
                    grad_y += np.tanh((y_pos - min_y) / alpha) - np.tanh((max_y - y_pos) / alpha)
                    
        return grad_x, grad_y
    
    def calculate_density_gradient(self, cell: Cell) -> Tuple[float, float]:
        """计算密度梯度"""
        cx, cy = cell.get_center()
        field_x, field_y = self.density_map.get_density_gradient(cx, cy)    
        # 单元的电荷等于面积
        cell_area = cell.get_area()    
        # 根据能量函数 N(v) = (1/2) * Σ qi*ψi(v)，梯度应该是 qi * 梯度(ψi)
        grad_x = field_x * cell_area
        grad_y = field_y * cell_area           
        return grad_x , grad_y

    def calculate_gradient(self, cell):
        """计算总梯度"""
        wl_grad_x, wl_grad_y = self.calculate_wirelength_gradient(cell)
        den_grad_x, den_grad_y = self.calculate_density_gradient(cell)

        # 总梯度 = beta * 线长梯度 + gamma * 密度梯度
        grad_x = self._beta * wl_grad_x + self._lambda * den_grad_x
        grad_y = self._beta * wl_grad_y + self._lambda * den_grad_y
        
        return grad_x, grad_y
    
    def update_learning_rate(self, iteration):
        """更新学习率"""
        # 随着迭代次数增加，逐渐减小学习率
        self.learning_rate = 1.0 * (1.0 - iteration / self.max_iterations)
        self.learning_rate = max(0.01, self.learning_rate)  # 确保学习率不会太小
        
    def clip_position(self, cell, x, y):
        """将单元位置限制在芯片区域内"""
        min_x, min_y, max_x, max_y = self.circuit.die_area
        x = max(min_x, min(max_x - cell.width, x))      # 确保不超出边界
        y = max(min_y, min(max_y - cell.height, y))
        return x, y

    def place(self):
        self.initialize()        # 初始化布局
        self.visualizer.visualize_placement(self.density_map, show_field=True,output_file=r"images/initial_placement.png",show=True)

        current_hpwl=0
        
        # 创建logs目录（如果不存在）
        os.makedirs("logs", exist_ok=True)

        # 全局布局迭代
        for iteration in range(self.max_iterations):
            self.update_learning_rate(iteration) # 更新学习率

            print(f"-----------------------iteration: {iteration}-----------------------")

            # 为当前迭代创建CSV文件
            log_file = f"logs/iteration_{iteration}.csv"
            with open(log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                # 写入表头
                writer.writerow([
                    "单元名称",
                    "当前位置X", "当前位置Y",
                    "线长梯度X", "线长梯度Y",
                    "密度梯度X", "密度梯度Y",
                    "加权密度梯度X", "加权密度梯度Y",
                    "总梯度X", "总梯度Y",
                    "更新后位置X", "更新后位置Y"
                ])

            cell_gradients_hpwl={} # 存储所有单元的线长梯度
            cell_gradients_density={} # 存储所有单元的密度梯度
            cell_gradients={} # 存储所有单元的总梯度
            max_movement = 0.0  # 对每个未固定的单元计算力并更新位置   

            for cell_name,cell in self.circuit.cells.items():
                grad_x_hpwl, grad_y_hpwl = self.calculate_wirelength_gradient(cell)
                grad_x_density, grad_y_density = self.calculate_density_gradient(cell)
                cell_gradients_hpwl[cell_name]=(grad_x_hpwl,grad_y_hpwl)
                cell_gradients_density[cell_name]=(grad_x_density,grad_y_density)

            total_grad_hpwl=sum((x**2+y**2)**0.5 for x,y in cell_gradients_hpwl.values())
            total_grad_density=sum((x**2+y**2)**0.5 for x,y in cell_gradients_density.values())

            max_density = self.density_map.get_max_density()  
            print(f"当前最大密度: {max_density}")


            last_hpwl=current_hpwl
            current_hpwl=self.circuit.get_total_hpwl()
            print(f"当前线长: {current_hpwl}")

            delta_hpwl=current_hpwl-last_hpwl
            print(f"当前线长变化: {delta_hpwl}")

            self.update_lambda(delta_hpwl,total_grad_hpwl,total_grad_density)
            print(f"当前lambda: {self._lambda}")

            current_energy=self.density_map.get_total_energy()
            print(f"当前能量: {current_energy}")
            print(f"加权后的能量: {current_energy*self._lambda}")

            print(f"目标函数值为: {current_energy*self._lambda+current_hpwl}")


            # print(f" 迭代中 Cell 的 Information ")
            for cell_name,cell in self.circuit.cells.items():
                if not cell.is_fixed and not cell.is_macro:
                    # 分别获取线长和密度梯度的x和y分量
                    grad_x_hpwl, grad_y_hpwl = cell_gradients_hpwl[cell_name]
                    grad_x_density, grad_y_density = cell_gradients_density[cell_name]
                    
                    # 计算总梯度
                    grad_x = grad_x_hpwl + self._lambda * grad_x_density
                    grad_y = grad_y_hpwl + self._lambda * grad_y_density

                    cell_gradients_density[cell_name]=(grad_x_hpwl,grad_y_hpwl)
                    cell_gradients[cell.name] = (grad_x, grad_y)

                    # 更新位置
                    new_x = cell.x - self.learning_rate * grad_x
                    new_y = cell.y - self.learning_rate * grad_y
                    
                    new_x, new_y = self.clip_position(cell, new_x, new_y)  # 限制在芯片区域内

                    # 记录单元信息到CSV文件
                    with open(log_file, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            cell_name,
                            f"{cell.x:.3f}", f"{cell.y:.3f}",
                            f"{grad_x_hpwl:.3f}", f"{grad_y_hpwl:.3f}",
                            f"{grad_x_density:.3f}", f"{grad_y_density:.3f}",
                            f"{grad_x_density * self._lambda:.3f}", f"{grad_y_density * self._lambda:.3f}",
                            f"{grad_x:.3f}", f"{grad_y:.3f}",
                            f"{new_x:.3f}", f"{new_y:.3f}"
                        ])
                    
                    # 计算移动距离
                    movement = np.sqrt((new_x - cell.x)**2 + (new_y - cell.y)**2)
                    max_movement = max(max_movement, movement)                    
                    cell.move_to(new_x, new_y)
            
            # 移动宏单元（通常宏单元移动频率较低）
            if iteration % 5 == 0:  # 每5次迭代更新一次宏单元
                for cell_name, cell in self.circuit.cells.items():
                    if not cell.is_fixed and cell.is_macro:
                        grad_x, grad_y = self.calculate_gradient(cell)
                        cell_gradients[cell.name] = (grad_x, grad_y)
                        
                        macro_lr = self.learning_rate * 0.2  # 对宏单元使用较小的学习率      
                       
                        new_x = cell.x - macro_lr * grad_x # 更新位置
                        new_y = cell.y - macro_lr * grad_y                        
                        
                        new_x, new_y = self.clip_position(cell, new_x, new_y)   # 限制在芯片区域内                     
                       
                        movement = np.sqrt((new_x - cell.x)**2 + (new_y - cell.y)**2) # 计算移动距离
                        max_movement = max(max_movement, movement)                      
                       
                        cell.move_to(new_x, new_y) # 更新位置
            
            # 更新密度图
            self.update_density_map()


            # 每Times次迭代更新一次可视化
            output_file=fr"images/placement_{iteration}.png"
            if iteration % TIMEs == 0:
                # 准备不同类型的梯度信息
                total_gradients = {}
                wirelength_gradients = {}
                weighted_density_gradients = {}
                
                for cell_name, cell in self.circuit.cells.items():
                    if not cell.is_fixed and not cell.is_macro:
                        # 获取线长梯度
                        grad_x_hpwl, grad_y_hpwl = cell_gradients_hpwl[cell_name]
                        wirelength_gradients[cell_name] = (grad_x_hpwl, grad_y_hpwl)
                        
                        # 获取加权密度梯度
                        grad_x_density, grad_y_density = cell_gradients_density[cell_name]
                        weighted_grad_x = grad_x_density * self._lambda
                        weighted_grad_y = grad_y_density * self._lambda
                        weighted_density_gradients[cell_name] = (weighted_grad_x, weighted_grad_y)
                        
                        # 获取总梯度
                        total_gradients[cell_name] = cell_gradients[cell_name]

                self.visualizer.visualize_placement(
                    self.density_map, 
                    show_field=True,
                    output_file=output_file,
                    gradients=total_gradients,
                    wirelength_gradients=wirelength_gradients,
                    weighted_density_gradients=weighted_density_gradients,
                    show=True
                )
                
                # self.visualizer.visualize_density(self.density_map,output_file=fr"images/density_{iteration}.png",show=False)
                # self.visualizer.visualize_potential(self.density_map,output_file=fr"images/potential_{iteration}.png",show=False)

                logger.info(f"迭代 {iteration}:能量={current_energy:.2f}, HPWL={current_hpwl:.2f}, "
                           f"最大密度={max_density:.2f}, "f"最大移动={max_movement:.2f}")
                
            
            # 收敛检查
            if max_movement < 0.1 and iteration > 20:
                logger.info(f"布局在第 {iteration} 次迭代后收敛")
                break
        
        # 显示最终布局
        # self.visualizer.visualize_placement(self.density_map, show_field=True,output_file=r"images/final_placement.png")
        
        # 最终评估
        final_energy=self.density_map.get_total_energy()
        final_hpwl = self.circuit.get_total_hpwl()
        final_density = self.density_map.get_max_density()
        logger.info(f"布局完成: 最终能量={final_energy:.2f}, 最终HPWL={final_hpwl:.2f}, 最终最大密度={final_density:.2f}")
      
    def legalize(self):
        """
        执行布局合法化(legalization)
        解决重叠问题，确保所有单元不重叠且对齐到行
        """
        logger.info("开始执行详细布局(legalization)")
        # 首先处理宏单元
        macro_cells = [cell for cell in self.circuit.cells.values() 
                      if cell.is_macro and not cell.is_fixed]
        
        # 根据面积从大到小排序
        macro_cells.sort(key=lambda c: c.get_area(), reverse=True)
        
        # 逐个放置宏单元
        for cell in macro_cells:
            self.legalize_macro(cell)
        
        # 然后处理标准单元
        std_cells = [cell for cell in self.circuit.cells.values() if not cell.is_macro and not cell.is_fixed]
        
        # (简化实现) 使用一种简单的贪心算法进行标准单元摆放
        std_cells.sort(key=lambda c: c.get_area(), reverse=True)
        
        # 行高和行位置（假设所有标准单元高度相同）
        row_height = 1.0  # 假设的行高
        min_x, min_y, max_x, max_y = self.circuit.die_area
        
        # 创建行
        num_rows = int((max_y - min_y) / row_height)
        self.circuit.rows = []
        for i in range(num_rows):
            row_y = min_y + i * row_height
            self.circuit.rows.append((min_x, row_y, max_x - min_x, row_height))
        
        # 逐行放置单元
        current_row = 0
        current_x = min_x
        
        for cell in std_cells:
            # 如果当前行放不下，换到下一行
            if current_x + cell.width > max_x:
                current_row = (current_row + 1) % num_rows
                current_x = min_x
            
            # 放置单元
            row_x, row_y, _, _ = self.circuit.rows[current_row]
            cell.move_to(current_x, row_y)
            
           # 更新当前位置
            current_x += cell.width
        
        logger.info(f"详细布局完成，所有单元已合法化")
    
    def legalize_macro(self, macro_cell: Cell):
        """对宏单元进行合法化"""
        # 找到当前位置附近的合法位置
        min_x, min_y, max_x, max_y = self.circuit.die_area
        
        # 当前位置
        current_x, current_y = macro_cell.x, macro_cell.y
        
        # 查找最近的合法位置（不与其他宏单元重叠）
        best_x, best_y = current_x, current_y
        min_overlap = float('inf')
        
        # 搜索范围
        search_step = macro_cell.width / 4
        search_range = macro_cell.width * 2
        
        # 在当前位置附近搜索
        for x_offset in np.arange(-search_range, search_range, search_step):
            for y_offset in np.arange(-search_range, search_range, search_step):
                test_x = current_x + x_offset
                test_y = current_y + y_offset
                
                # 确保在芯片区域内
                if test_x < min_x or test_x + macro_cell.width > max_x or \
                   test_y < min_y or test_y + macro_cell.height > max_y:
                    continue
                
                # 检查与其他宏单元的重叠
                overlap_area = 0
                for other_cell in self.circuit.cells.values():
                    if other_cell.name == macro_cell.name or not other_cell.is_macro:
                        continue
                    
                    # 计算重叠区域
                    x_overlap = max(0, min(test_x + macro_cell.width, other_cell.x + other_cell.width) - max(test_x, other_cell.x))
                    y_overlap = max(0, min(test_y + macro_cell.height, other_cell.y + other_cell.height) - max(test_y, other_cell.y))
                    overlap_area += x_overlap * y_overlap
                

                # 更新最佳位置
                if overlap_area < min_overlap:
                    min_overlap = overlap_area
                    best_x, best_y = test_x, test_y
                    
                    # 如果没有重叠，立即返回
                    if overlap_area == 0:
                        break
            
            # 如果找到无重叠位置，立即返回
            if min_overlap == 0:
                break
        
        # 移动宏单元到最佳位置
        macro_cell.move_to(best_x, best_y) # 更新当前位置