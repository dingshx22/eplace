"""
基于静电力学的布局器实现
"""
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
from src.circuit import Circuit, Cell
from src.density import DensityMap
from src.visualizer import PlacementVisualizer
import time
import os
import csv
from src.utility import NesterovOptimizer, estimate_lipschitz_constant

logger = logging.getLogger("ePlace.Placer")

TIMEs = 1

HPWL_ref = 3.5e5
MU_0 = 1.1

COF_max = 1.05
COF_min = 0.95


class ElectrostaticPlacer:

    def __init__(self,circuit: Circuit,alpha=1.0,max_iterations=100,bin_size=5):
        self.circuit = circuit
        self.alpha = alpha         # 静电力学模型参数
        self._beta = 1             # 线长权重
        self._lambda = 0           # 密度权重
        self.max_iterations = max_iterations
        self.bin_size = bin_size
        self.x0 = []

        # 初始化密度图
        self.density_map = DensityMap(self.circuit.die_area[0], self.circuit.die_area[1], \
                                    self.circuit.die_area[2], self.circuit.die_area[3], bin_size=self.bin_size)

        self.learning_rate = 0.1
        self.visualizer = PlacementVisualizer(self.circuit)

    def update_loaction(self, x: np.ndarray):
        for i, cell in enumerate(self.circuit.cells.values()):
            if not cell.is_fixed:
                cell.x, cell.y = self.clip_position(cell, x[2 * i], x[2 * i + 1])

    def objective_function(self, x: np.ndarray):  #目标函数

        # #更新单元位置
        # self.update_loaction(x)
        # self.update_density_map()

        hpwl = self.circuit.get_total_hpwl()
        nv = self.density_map.get_total_energy()
        return hpwl + self._lambda * nv

    def gradient_function(self, x: np.ndarray) -> np.ndarray:  #梯度函数
        grad = np.zeros_like(x)

        # # 更新单元位置
        # self.update_loaction(x)
        # self.update_density_map()  # 取消注释，确保密度图更新

        # 计算每个单元的梯度
        for i, cell in enumerate(self.circuit.cells.values()):
            if not cell.is_fixed:
                w_grad_x, w_grad_y = self.calculate_wirelength_gradient(cell)  # 计算线长梯度
                d_grad_x, d_grad_y = self.calculate_density_gradient(cell)  # 计算密度梯度
                grad[2 * i] = w_grad_x + self._lambda * d_grad_x  # 总梯度
                grad[2 * i + 1] = w_grad_y + self._lambda * d_grad_y
        return grad

    def initialize(self):
        min_x, min_y, max_x, max_y = self.circuit.die_area
        for _, cell in self.circuit.cells.items():  # 初始随机布局（对于未固定的单元）
            if not cell.is_fixed:
                if cell.is_macro:
                    # 大型宏单元放在边缘
                    edge_margin = 50
                    max_x = max_x - cell.width - edge_margin
                    max_y = max_y - cell.height - edge_margin
                    min_x_loc = min_x + edge_margin
                    min_y_loc = min_y + edge_margin
                else:
                    min_x_loc = min_x  # 标准单元随机分布
                    min_y_loc = min_y

                cell.x = np.random.uniform(min_x_loc, max_x - cell.width)
                cell.y = np.random.uniform(min_y_loc, max_y - cell.height)

                self.x0.extend([cell.x, cell.y])
        self.x0 = np.array(self.x0)  # 准备初始位置向量
        self.update_density_map()  # 初始化密度图

        fenzi = 0
        fenmu = 0
        for _, cell in self.circuit.cells.items():
            wx, wy = self.calculate_wirelength_gradient(cell)
            # print(f"线长梯度={wx}，{wy}")
            ksix, ksiy = self.calculate_density_gradient(cell)
            # print(f"密度梯度={ksix}，{ksiy}")


            fenzi += np.abs(wx) + np.abs(wy)
            fenmu += (np.abs(ksix) + np.abs(ksiy)) * cell.get_area()

        self._lambda = fenzi / fenmu  # 初始化密度权重
        print(f"初始化,即随机放置完成")
        print(f"初始化线长梯度={fenzi}，初始化密度梯度={fenmu}")
        print(f"初始化密度权重={self._lambda}")

    def update_density_map(self):
        self.density_map.clear()
        for _, cell in self.circuit.cells.items():
            self.density_map.add_cell(cell)

    def update_lambda(self, delta_hpwl):
        p = delta_hpwl / HPWL_ref + 1
        mu_k = pow(MU_0, p)
        mu_k = max(mu_k, 0.75)
        mu_k = min(mu_k, 1.1)
        self._lambda = self._lambda * mu_k
        print(f"更新密度权重={self._lambda}")

    def calculate_wirelength_gradient_WA(self, cell: Cell, gamma=1.0):
        grad_x, grad_y = 0.0, 0.0
    
        related_nets = []
        for pin in cell.pins.values():
            if pin.parent_net:
                related_nets.extend(pin.parent_net)
        
        # Print count for debugging
        # print(f"Cell {cell.name} is connected to {len(related_nets)} nets")
        
        # For each related network, calculate gradients
        for net in related_nets:
            if len(net.pins) <= 1:
                continue
                
            positions = [pin.absolute_position for pin in net.pins]
            if not positions:
                continue
            xs, ys = zip(*positions)
            
            # Calculate gradients for pins in this cell only
            cell_pins = [pin for pin in net.pins if pin.parent_cell.name == cell.name]
            
            for pin in cell_pins:
                x_pos, y_pos = pin.absolute_position
                
                # x direction gradient (WA model)
                x_grad = 0
                for x_j in xs:
                    if abs(x_pos - x_j) > 1e-6:  # Skip self-comparison
                        dx_plus = gamma + x_pos - x_j
                        dx_minus = gamma - x_pos + x_j
                        x_grad += (1.0 / max(dx_plus, 1e-6)) - (1.0 / max(dx_minus, 1e-6))
                

                y_grad = 0
                for y_j in ys:
                    if abs(y_pos - y_j) > 1e-6:  # Skip self-comparison
                        dy_plus = gamma + y_pos - y_j
                        dy_minus = gamma - y_pos + y_j
                        y_grad += (1.0 / max(dy_plus, 1e-6)) - (1.0 / max(dy_minus, 1e-6))
                
                
                scaling_factor = len(xs) * gamma  # Normalize by net size and gamma
                grad_x += x_grad / scaling_factor
                grad_y += y_grad / scaling_factor
        
        return grad_x, grad_y
    
    def calculate_wirelength_gradient(self, cell: Cell):
        grad_x, grad_y = 0.0, 0.0

        related_nets = []
        for pin in cell.pins.values():
            if pin.parent_net:
                related_nets.extend(pin.parent_net)
        
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

    def calculate_density_gradient(self, cell: Cell):
        cx, cy = cell.get_center()
        field_x, field_y = self.density_map.get_density_gradient(cx, cy)
        # 单元的电荷等于面积
        cell_area = cell.get_area()
        # 根据能量函数 N(v) = (1/2) * Σ qi*ψi(v)，梯度应该是 qi * 梯度(ψi)
        grad_x = field_x * cell_area
        grad_y = field_y * cell_area
        return grad_x, grad_y

    def calculate_gradient(self, cell):
        wl_grad_x, wl_grad_y = self.calculate_wirelength_gradient(cell)
        den_grad_x, den_grad_y = self.calculate_density_gradient(cell)

        # 总梯度 = beta * 线长梯度 + gamma * 密度梯度
        grad_x = self._beta * wl_grad_x + self._lambda * den_grad_x
        grad_y = self._beta * wl_grad_y + self._lambda * den_grad_y

        return grad_x, grad_y

    def clip_position(self, cell, x, y):
        min_x, min_y, max_x, max_y = self.circuit.die_area
        x = max(min_x, min(max_x - cell.width, x))  # 确保不超出边界
        y = max(min_y, min(max_y - cell.height, y))
        return x, y

    def place(self):
        self.initialize()  # 初始化布局
        self.visualizer.visualize_placement(self.density_map,show_field=True,output_file=r"images/initial_placement.png",show=True)

        os.makedirs("logs", exist_ok=True)  # 创建logs目录（如果不存在）
        L = 10
        mu = 0.9

        x_now = self.x0.copy()
        x_prev = x_now.copy()

        # 初始化梯度
        grad_now = self.gradient_function(x_now)
        grad_prev = np.zeros_like(grad_now)  # 初始化为0向量而不是复制

        # 初始化线长
        hpwl_now = self.circuit.get_total_hpwl()
        hpwl_prev = 0  # 初始化为0而不是复制

        v = np.zeros_like(x_now)
        history = []

        for it, _ in enumerate(range(self.max_iterations)):
            print(f"--------------当前迭代次数：{it+1}------------------")


            self.visualizer.visualize_placement(
                self.density_map,
                show_field=True,
                output_file=fr"images/placement_{it}.png",
                show=True)
            
            
            self.update_lambda(delta_hpwl = hpwl_now - hpwl_prev)


            # 计算动量项
            y = x_now + mu * v
            _grad = self.gradient_function(y)

            # 更新Lipschitz常数
            if it > 0:
                denominator = np.linalg.norm(x_now - x_prev)
                print(f"坐标的变化为={denominator}")
                print(f"梯度的变化为={np.linalg.norm(grad_now - grad_prev)}")

                if denominator < 1e-10:
                    L = max(L, 1.0)  # 使用一个合理的下界
                else:
                    L = np.linalg.norm(grad_now - grad_prev) / denominator
                L = max(L, 1)  # 确保L不会太小

            print(f"Lipschitz常数={L}")

            # 更新动量和位置
            v = mu * v - (1 / L) * _grad
            x_prev = x_now.copy()
            grad_prev = grad_now.copy()
            x_now = x_now + v  # 使用x_prev来更新x_now
            grad_now = self.gradient_function(x_now)

            print("x_now变化量:", np.linalg.norm(x_now - x_prev))

            # 更新位置和密度图
            self.update_loaction(x_now)
            self.update_density_map()

            hpwl_prev = hpwl_now            
            hpwl_now = self.circuit.get_total_hpwl()


            # 打印当前状态
            print(f"当前能量={self._lambda*self.density_map.get_total_energy()}")
            print(f"当前线长={self.circuit.get_total_hpwl()}")
            print(f"当前梯度范数={np.linalg.norm(_grad)}")  # 添加梯度范数的打印

            history.append(self.objective_function(x_now))

            if np.linalg.norm(x_now - x_prev) < 1e-6:
                break

            print("------------------------------------")

        # L = estimate_lipschitz_constant(self.gradient_function, self.x0)        # 估计Lipschitz常数
        # optimizer = NesterovOptimizer(f=self.objective_function,grad_f=self.gradient_function,L=L,x0=self.x0)
        # x_opt, history = optimizer.optimize()

        # # 更新最终位置
        # for i, cell in enumerate(self.circuit.cells.values()):
        #     if not cell.is_fixed:
        #         cell.x = x_opt[2*i]
        #         cell.y = x_opt[2*i+1]

        # 更新密度图
        self.update_density_map()

        # 显示最终布局
        self.visualizer.visualize_placement(
            self.density_map,
            show_field=True,
            output_file=r"images/final_placement.png",
            show=True)

        # 最终评估
        final_energy = self.density_map.get_total_energy()
        final_hpwl = self.circuit.get_total_hpwl()
        final_density = self.density_map.get_max_density()
        logger.info(
            f"布局完成: 最终能量={final_energy:.2f}, 最终HPWL={final_hpwl:.2f}, 最终最大密度={final_density:.2f}"
        )

    def legalize(self):
        """
        执行布局合法化(legalization)
        解决重叠问题，确保所有单元不重叠且对齐到行
        """
        logger.info("开始执行详细布局(legalization)")
        # 首先处理宏单元
        macro_cells = [
            cell for cell in self.circuit.cells.values()
            if cell.is_macro and not cell.is_fixed
        ]

        # 根据面积从大到小排序
        macro_cells.sort(key=lambda c: c.get_area(), reverse=True)

        # 逐个放置宏单元
        for cell in macro_cells:
            self.legalize_macro(cell)

        # 然后处理标准单元
        std_cells = [
            cell for cell in self.circuit.cells.values()
            if not cell.is_macro and not cell.is_fixed
        ]

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
            for y_offset in np.arange(-search_range, search_range,
                                      search_step):
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
                    x_overlap = max(
                        0,
                        min(test_x + macro_cell.width, other_cell.x +
                            other_cell.width) - max(test_x, other_cell.x))
                    y_overlap = max(
                        0,
                        min(test_y + macro_cell.height, other_cell.y +
                            other_cell.height) - max(test_y, other_cell.y))
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
        macro_cell.move_to(best_x, best_y)  # 更新当前位置
