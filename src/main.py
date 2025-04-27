import src.configs as configs
import time
from src.circuit import Circuit
from src.placer import ElectrostaticPlacer
from src.visualizer import PlacementVisualizer
from src.density import DensityMap
import numpy as np


def main():   #主函数
    args = configs.parse_arguments()
    logger = configs.setup_logger()

    logger.info("启动ePlace布局算法 ")
    start_time = time.time()

    circuit = Circuit() # 创建电路
    circuit.load_from_lef_def(args.lef, args.def_file, test=True)  # 默认使用测试模式
    logger.info(f"读取LEF文件{args.lef}和DEF文件{args.def_file} ，成功加载电路 ")
    cell_num = len(circuit.cells)
    bin_size = np.ceil(np.log2(np.sqrt(cell_num)))


    # visualizer = PlacementVisualizer(circuit) if args.visualize else None  # 创建可视化器

    # 创建布局器
    placer = ElectrostaticPlacer(circuit=circuit,alpha=args.alpha,max_iterations=args.iterations, bin_size=bin_size)

    logger.info("开始执行布局")
    placer.place()


    end_time = time.time()
    logger.info(f"布局完成，耗时: {end_time - start_time:.2f}秒")
    circuit.print_area_stats()




if __name__ == "__main__":
    main()