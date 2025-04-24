import src.configs as configs
import time
from src.circuit import Circuit
from src.placer import ElectrostaticPlacer
from src.visualizer import PlacementVisualizer
from src.density import DensityMap


def main():   #主函数
    args = configs.parse_arguments()
    logger = configs.setup_logger()
    
    logger.info("启动ePlace布局算法 ")
    start_time = time.time()
    
    # 加载电路

    circuit = Circuit()
    circuit.load_from_lef_def(args.lef, args.def_file, test=True)  # 默认使用测试模式
    logger.info(f"读取LEF文件{args.lef}和DEF文件{args.def_file} ，成功加载电路 ")   

    # 创建可视化器

    visualizer = PlacementVisualizer(circuit) if args.visualize else None
    
    # 创建布局器
    placer = ElectrostaticPlacer(
        circuit=circuit,
        alpha=args.alpha,
        max_iterations=args.iterations,
        bin_size=4
    )
    
    # 执行布局
    logger.info("开始执行布局")
    placer.place()
    
    # 保存结果
    # logger.info(f"保存布局结果到{args.output}")
    # circuit.save_to_def(args.output)
    
    
    end_time = time.time()
    logger.info(f"布局完成，耗时: {end_time - start_time:.2f}秒")
    circuit.print_area_stats()
    print("ok")



if __name__ == "__main__":
    main()