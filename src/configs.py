import argparse
import logging
import sys

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='ePlace')
    
    # 必需参数
    parser.add_argument('--lef',    type=str,  default='dummy.lef',  required=False,   help='LEF文件路径')
    parser.add_argument('--def',    dest='def_file', type=str,    default='dummy.def', required=False, help='DEF文件路径')
    parser.add_argument('--output', type=str,   default='output.def', required=False, help='输出DEF文件路径')
    
    
    # 可选参数
    parser.add_argument('--alpha', type=float, default=1.0, help='静电力学模型参数')
    parser.add_argument('--beta', type=float, default=0.5, help='线长权重参数')
    parser.add_argument('--gamma', type=float, default=0.5, help='密度权重参数')
    parser.add_argument('--iterations', type=int, default=10, help='全局布局迭代次数')
    # parser.add_argument('--visualize', action='store_true', help='是否可视化布局结果')
    parser.add_argument('--visualize', type=bool, default=True, help='是否可视化布局结果')
    
    return parser.parse_args()


def setup_logger():
    """设置日志配置"""
    logger = logging.getLogger("ePlace")
    logger.setLevel(logging.INFO)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # 设置日志格式
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)

    # 添加处理器到logger
    logger.addHandler(console_handler)
    
    return logger 