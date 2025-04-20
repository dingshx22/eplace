# ePlace-MS: Electrostatics-Based Placement for Mixed-Size Circuits

本项目是对论文 "ePlace-MS: Electrostatics-Based Placement for Mixed-Size Circuits" 中所提出算法的Python实现。这个算法利用静电力学模型进行混合尺寸集成电路的布局优化。

## 项目概述

ePlace-MS算法的关键思想是将电路布局问题建模为静电力学问题：
1. 将电路单元视为带电粒子
2. 使用静电场模型来解决布局中的拥塞问题
3. 结合线长和密度优化目标

该算法特别适合处理混合尺寸的电路，即同时包含标准单元和大型宏单元的电路。

## 项目结构

```
eplace-ms/
├── src/
│   ├── __init__.py
│   ├── main.py            # 主程序入口
│   ├── circuit.py         # 电路模型
│   ├── density_map.py     # 密度图实现
│   ├── placer.py          # 静电力布局器
│   └── visualizer.py      # 布局可视化
├── examples/              # 示例电路
├── setup.py               # 安装配置
└── README.md              # 项目说明
```

## 安装

```bash
# 克隆仓库
git clone https://github.com/yourusername/eplace-ms.git
cd eplace-ms

# 安装依赖
pip install -e .
```

## 使用方法

```bash
# 基本用法
eplace-ms --lef <lef_file> --def <def_file> --output <output_def_file>

# 完整选项
eplace-ms --lef input.lef --def input.def --output output.def --iterations 200 --alpha 1.0 --beta 0.5 --gamma 0.5 --visualize
```

### 参数说明

- `--lef`: LEF文件路径
- `--def`: DEF文件路径
- `--output`: 输出DEF文件路径
- `--iterations`: 全局布局迭代次数
- `--alpha`: 静电力学模型中的alpha参数
- `--beta`: 线长权重参数
- `--gamma`: 密度权重参数
- `--visualize`: 是否可视化布局结果

## 算法核心

ePlace-MS算法的核心步骤包括：

1. **初始化布局**：为所有单元分配初始位置
2. **全局布局**：
   - 计算线长梯度（使用对数-求和模型LSE）
   - 计算密度梯度（基于静电场模型）
   - 更新单元位置（使用梯度下降）
3. **详细布局**：解决单元重叠问题，将单元对齐到行
4. **评估布局质量**：计算半周长线长(HPWL)和密度分布

## 可视化

该项目提供两种可视化方式：
1. 布局图：显示所有单元在芯片上的位置
2. 密度热图：显示单元密度分布

## 论文参考

本项目基于以下论文实现：

> J. Lu, P. Chen, C.-C. Chang, L. Sha, D. J.-H. Huang, C.-C. Teng, and C.-K. Cheng, "ePlace: Electrostatics based placement using nesterov's method," in Proceedings of the 51st Annual Design Automation Conference, 2014, pp. 1–6.

和其混合尺寸电路扩展：

> J. Lu, H. Zhuang, P. Chen, H. Chang, C.-C. Chang, Y.-C. Wong, L. Sha, D. Huang, Y. Luo, C.-C. Teng, and C.-K. Cheng, "ePlace-MS: Electrostatics-based placement for mixed-size circuits," IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems, vol. 34, no. 5, pp. 685–698, 2015.

## 性能注意事项

- 对于大型电路，密度图的网格大小可能需要调整以平衡精度和计算效率
- 实际应用中，可能需要利用多线程或GPU加速计算密度梯度

## 许可

MIT