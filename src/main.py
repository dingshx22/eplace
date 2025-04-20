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

    density_map = DensityMap(circuit.die_area[0], circuit.die_area[1], circuit.die_area[2], circuit.die_area[3],grid_size=10)


    # 创建可视化器
    
    visualizer = PlacementVisualizer(circuit) if args.visualize else None
    visualizer.visualize(density_map=density_map, show_field=True,output_file="initial_placement.png")

    
    # 创建布局器
    placer = ElectrostaticPlacer(
        circuit=circuit,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        max_iterations=args.iterations
    )
    
    # 执行布局
    logger.info("开始执行布局")
    placer.place()
    
    # 保存结果
    # logger.info(f"保存布局结果到{args.output}")
    # circuit.save_to_def(args.output)
    
    # 可视化最终结果
    if args.visualize:
        visualizer.visualize(output_file="final_placement.png")
    
    end_time = time.time()
    logger.info(f"布局完成，耗时: {end_time - start_time:.2f}秒")
    circuit.print_area_stats()



if __name__ == "__main__":
    main()