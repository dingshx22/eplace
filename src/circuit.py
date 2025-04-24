"""
电路和元件模型类
"""

import numpy as np
import logging
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger("ePlace.Circuit")

STD_NUM = 80
MACRO_NUM = 0
# NET_NUM = 0

DIE_WIDTH = 20
DIE_HEIGHT = 20



CELL_LIB = {"STD_CELLS":{
            "NAND2":  {"width": 1.0, "height": 1.0,   "pins":   {"A": (0.2, 0.5), "B": (0.5, 0.5), "Y": (0.8, 0.5) }},
            "NOR2":   {"width": 1.0, "height": 1.0,   "pins":   {"A": (0.2, 0.5), "B": (0.5, 0.5), "Y": (0.8, 0.5) }},
            "INV":    {"width": 0.5, "height": 1.0,   "pins":   {"A": (0.2, 0.5), "Y": (0.4, 0.5)                  }},
            "DFF":    {"width": 2.0, "height": 1.0,   "pins":   {"D": (0.2, 0.5), "CK": (0.5, 0.5), "Q": (1.8, 0.5)}}
            }, 
            "MACROS":{
            "MACRO1": {"width": 10.0, "height": 10.0, "pins":   {"A": (5.0, 0.0), "Y": (5.0, 10.0)                 }}
            }
        }



class Pin:  #引脚类

    def __init__(self, name: str, parent_cell: 'Cell', offset_x: float, offset_y: float):
        self.name = name
        self.parent_cell = parent_cell
        self.parent_net = []
        self.offset_x = offset_x
        self.offset_y = offset_y


    @property
    def absolute_position(self) -> Tuple[float, float]:  #获取引脚的绝对位置
        if self.parent_cell:
            return (self.parent_cell.x + self.offset_x,
                    self.parent_cell.y + self.offset_y)
        return (self.offset_x, self.offset_y)

class Cell:

    def __init__(self, name, width, height, is_macro=False, is_fixed=False):
        self.name = name
        self.width = width
        self.height = height
        self.is_macro = is_macro
        self.is_fixed = is_fixed
        self.x = 0.0
        self.y = 0.0
        self.pins: Dict[str, Pin] = {}

    def add_pin(self, name, offset_x, offset_y):  # 添加引脚
        pin = Pin(name, self, offset_x, offset_y)
        self.pins[name] = pin
        return pin

    def get_area(self) -> float:  # 获取单元面积
        return self.width * self.height

    def get_center(self) -> Tuple[float, float]:
        return (self.x + self.width / 2, self.y + self.height / 2)

    def is_overlaps_with(self, other: "Cell") -> bool:  # 检查是否与其他单元重叠
        return not (self.x + self.width <= other.x or other.x + other.width
                    <= self.x or self.y + self.height <= other.y
                    or other.y + other.height <= self.y)

    def move_to(self,x,y):  # 移动单元到指定位置
        if not self.is_fixed:
            self.x = x
            self.y = y
        else:
            logger.warning(f"单元{self.name}是固定单元，不能移动")

class Net:  # 网络类

    def __init__(self, name: str):
        self.name = name
        self.pins: List[Pin] = []

    def add_pin(self, pin: Pin):  # 添加引脚到网表
        self.pins.append(pin)
        pin.parent_net.append(self)  # 将当前网络添加到引脚的 parent_net 列表中

    def get_bounding_box(self):  # 得到 bounding box (min_x, min_y, max_x, max_y)
        if not self.pins:
            return (0, 0, 0, 0)

        positions = [pin.absolute_position for pin in self.pins]
        xs, ys = zip(*positions)
        return (min(xs), min(ys), max(xs), max(ys))

    def get_hpwl(self):
        min_x, min_y, max_x, max_y = self.get_bounding_box()
        return (max_x - min_x) + (max_y - min_y)



class Circuit:

    def __init__(self):
        self.cells: Dict[str, Cell] = {}
        self.nets: Dict[str, Net] = {}
        self.rows: List[Tuple[float, float, float,float]] = []  # 布局行 (x, y, width, height)
        self.die_area: Tuple[float, float, float,float] = (0, 0, 0, 0)  # (min_x, min_y, max_x, max_y)

    def add_cell(self, cell: Cell):
        self.cells[cell.name] = cell

    def get_std_cell(self):
        return [cell for cell in self.cells.values() if not cell.is_macro]

    def get_macro_cell(self):
        return [cell for cell in self.cells.values() if cell.is_macro]

    def add_net(self, net: Net):
        self.nets[net.name] = net

    def set_die_area(self, min_x: float, min_y: float, max_x: float,
                     max_y: float):  #设置芯片区域
        self.die_area = (min_x, min_y, max_x, max_y)

    def get_die_width(self):
        return self.die_area[2] - self.die_area[0]

    def get_die_height(self):
        return self.die_area[3] - self.die_area[1]

    def get_total_hpwl(self):
        return sum(net.get_hpwl() for net in self.nets.values())

    def load_from_lef_def(self, lef_path: str = None, def_path: str = None, test: bool = True):  #从LEF和DEF文件加载电路
        logger.info("解析LEF和DEF文件")
        if test:
            logger.info("使用测试数据")
            cell_lib = self._parse_lef(lef_path)  # 使用内置的测试单元库
            self._parse_def(cell_lib,test=True)  # 使用内置的测试电路实例
        else:
            # 实际的LEF/DEF解析逻辑
            logger.error("目前只支持测试模式,请设置test=True")
            raise NotImplementedError("实际的LEF/DEF解析尚未实现")

        logger.info(f"电路加载完成: {len(self.cells)}个单元, {len(self.nets)}个网络")

    def init_random_placement(self, cell_lib: Dict[str, Dict], std_num: int,macro_num: int):  # 随机创建一些示例单元
        for i in range(std_num):
            if cell_lib:
                cell_type = np.random.choice(list(cell_lib["STD_CELLS"].keys()))
                lib_info = cell_lib["STD_CELLS"][cell_type]
            else:
                logger.error("单元库为空,请设置cell_lib,先暂时使用内置的测试单元库!")
                cell_type = np.random.choice(["NAND2", "NOR2", "INV", "DFF"])
                lib_info = cell_lib[cell_type]

            # is_fixed = np.random.random() < 0.1  # 10%概率为固定单元
            is_fixed = False
            cell = Cell(
                name=f"{cell_type}_{i}",
                width=lib_info["width"],
                height=lib_info["height"],
                is_macro=False,
                is_fixed=is_fixed,
            )
            # 设置初始位置
            cell.x = np.random.uniform(0, self.get_die_width() - cell.width)
            cell.y = np.random.uniform(0, self.get_die_height() - cell.height)

            # 添加引脚
            for pin_name, (offset_x, offset_y) in lib_info["pins"].items():
                cell.add_pin(pin_name, offset_x, offset_y)
            self.add_cell(cell)

        # 添加一些宏单元
        for i in range(macro_num):
            if cell_lib:
                macro_type = np.random.choice(list(cell_lib["MACROS"].keys()))
                lib_info = cell_lib["MACROS"][macro_type]
            else:
                logger.error("单元库为空,请设置cell_lib,先暂时使用内置的测试单元库!")
                macro_type = np.random.choice(["MACRO1"])
                lib_info = cell_lib["MACROS"][macro_type]

            cell = Cell(name=f"{macro_type}_{i}",
                        width=lib_info["width"],
                        height=lib_info["height"],
                        is_macro=True,
                        is_fixed=False)
            # 宏单元初始位置
            cell.x = np.random.uniform(0, self.get_die_width() - cell.width)
            cell.y = np.random.uniform(0, self.get_die_height() - cell.height)

            # 添加引脚
            for pin_name, (offset_x, offset_y) in lib_info["pins"].items():
                cell.add_pin(pin_name, offset_x, offset_y)
            self.add_cell(cell)

        # 创建一些示例网表，确保每个单元至少连接到一个网络
        cells_list = list(self.cells.values())
        net_num = int(len(cells_list) * 1.5)  # 网络数量设置为单元数量的1.5倍

        # 第一轮：确保每个单元都至少连接到一个网络
        unconnected_cells = cells_list.copy()
        net_idx = 0

        while unconnected_cells:
            net = Net(f"net_{net_idx}")
            # 从未连接的单元中选择1个，从所有单元中选择1-3个
            main_cell = unconnected_cells.pop(0)
            num_additional = np.random.randint(1, 4)  # 额外选择1-3个单元
            additional_cells = np.random.choice(cells_list, size=min(num_additional, len(cells_list)), replace=False)

            # 连接主单元
            if main_cell.pins:
                pin_name = np.random.choice(list(main_cell.pins.keys()))
                net.add_pin(main_cell.pins[pin_name])

            # 连接额外的单元
            for cell in additional_cells:
                if cell.pins:
                    pin_name = np.random.choice(list(cell.pins.keys()))
                    net.add_pin(cell.pins[pin_name])

            self.add_net(net)
            net_idx += 1

        # 第二轮：添加剩余的网络
        remaining_nets = net_num - net_idx
        for i in range(remaining_nets):
            net = Net(f"net_{net_idx + i}")
            # 随机选择2-4个单元连接
            connected_cells = np.random.choice(cells_list,
                                               size=np.random.randint(2, 5),
                                               replace=False)
            for cell in connected_cells:
                if cell.pins:
                    pin_name = np.random.choice(list(cell.pins.keys()))
                    net.add_pin(cell.pins[pin_name])
            self.add_net(net)

    def _parse_lef(self, lef_path: str=None ,test: bool = True) -> Dict[str, Dict]:  # 解析LEF文件获取单元库信息
        if test:
            cell_lib = CELL_LIB
        else:
            # 实际的LEF解析逻辑
            logger.error("目前只支持测试模式，请设置test=True")
            raise NotImplementedError("实际的LEF解析尚未实现")
        return cell_lib

    def _parse_def(self , cell_lib: Dict[str, Dict], def_path: str=None, test: bool = True):  #解析DEF文件获取电路实例
        if test:
            # 设置芯片区域
            self.set_die_area(0, 0, DIE_WIDTH, DIE_HEIGHT)
            logger.info(f"芯片区域设置为: {self.die_area}")
            self.init_random_placement(cell_lib,std_num=STD_NUM,macro_num=MACRO_NUM)
            logger.info( f"电路实例创建完成: {len(self.cells)}个单元, ,{len(self.nets)}个网络")
        else:
            # 实际的DEF解析逻辑
            logger.error("目前只支持测试模式，请设置test=True")
            raise NotImplementedError("实际的DEF解析尚未实现")

    def save_to_def(self, def_path: str):  #保存布局结果到DEF文件
        logger.info(f"保存布局结果到: {def_path}")
        with open(def_path, 'w') as f:
            f.write("VERSION 5.8 ;\n")
            f.write("DESIGN circuit ;\n")

            # 写入芯片区域
            f.write(f"DIEAREA ( {self.die_area[0]} {self.die_area[1]} ) ( {self.die_area[2]} {self.die_area[3]} ) ;\n")

            # 写入单元位置
            f.write("COMPONENTS {0} ;\n".format(len(self.cells)))
            for cell_name, cell in self.cells.items():
                f.write(f"    - {cell_name} + PLACED ( {cell.x:.2f} {cell.y:.2f} ) N ;\n")
            f.write("END COMPONENTS\n")

            # 写入网表信息
            f.write("NETS {0} ;\n".format(len(self.nets)))
            for net_name, net in self.nets.items():
                f.write(f"    - {net_name}\n")
                for pin in net.pins:
                    f.write(f"        ( {pin.parent_cell.name} {pin.name} )\n")
                f.write("        ;\n")
            f.write("END NETS\n")

            f.write("END DESIGN\n")

    def calculate_area_stats(self) -> Dict:
        """计算电路中各类单元的面积统计"""

        stats = {
            "total_area": 0.0,
            "std_cell_area": 0.0,
            "macro_area": 0.0,
            "fixed_area": 0.0,
            "cell_count": {
                "total": len(self.cells),
                "std_cells": 0,
                "macros": 0,
                "fixed": 0,
            },
        }

        # 计算各类单元的面积
        for cell in self.cells.values():

            cell_area = cell.get_area()
            stats["total_area"] += cell_area
            if cell.is_fixed:
                stats["fixed_area"] += cell_area
                stats["cell_count"]["fixed"] += 1

            if cell.is_macro:
                stats["macro_area"] += cell_area
                stats["cell_count"]["macros"] += 1
            else:
                stats["std_cell_area"] += cell_area
                stats["cell_count"]["std_cells"] += 1

        # 计算芯片总面积和利用率
        die_area = self.get_die_width()*self.get_die_height()

        stats["die_area"] = die_area
        stats["utilization"] = stats["total_area"] / die_area  if die_area > 0 else 0

        # 添加百分比信息
        if stats["total_area"] > 0:
            stats["area_percentage"] = {
                "std_cells":stats["std_cell_area"] /  stats["total_area"] * 100,
                "macros":   stats["macro_area"]    /  stats["total_area"] * 100,
                "fixed":    stats["fixed_area"]    /  stats["total_area"] * 100,
            }

        return stats

    def print_area_stats(self):
        """打印电路面积统计信息"""
        stats = self.calculate_area_stats()

        # logger.info("  面积信息  :")
        logger.info(f"芯片面积: {stats['die_area']:.2f}")
        logger.info(f"单元面积: {stats['total_area']:.2f}")
        logger.info(f"芯片利用率: {stats['utilization']*100:.2f}%")

        logger.info(f"标准单元面积: {stats['std_cell_area']:.2f} ({stats['area_percentage']['std_cells']:.1f}%)")
        logger.info(f"宏单元面积: {stats['macro_area']:.2f} ({stats['area_percentage']['macros']:.1f}%)")
        logger.info(f"固定单元面积: {stats['fixed_area']:.2f} ({stats['area_percentage']['fixed']:.1f}%)")


        # logger.info(" 单元信息  :")
        logger.info(f"总单元数: {stats['cell_count']['total']}")
        logger.info(f"标准单元: {stats['cell_count']['std_cells']}")
        logger.info(f"宏单元:   {stats['cell_count']['macros']}")
        logger.info(f"固定单元: {stats['cell_count']['fixed']}")


        return stats
