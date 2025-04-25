import matplotlib.pyplot as plt

# def on_mouse_move(event):
#     if event.inaxes is not None:
#         print(f"鼠标在轴内，位置: x={event.xdata}, y={event.ydata}")
#     else:
#         print("鼠标不在轴内")

# # 创建图表和轴
# fig, ax = plt.subplots()
# ax.plot([0, 1, 2], [0, 1, 4])  # 示例数据

# # 绑定鼠标移动事件
# fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)

# # 显示图表
# plt.show()

# class PlotWithTooltip:
#     def __init__(self):
#         self.fig, self.ax = plt.subplots()
#         self.ax.plot([0, 1, 2], [0, 1, 4], marker='o')  # 示例数据
        
#         # 创建 tooltip
#         self.tooltip = self.ax.text(0, 0, '', bbox=dict(facecolor='white', alpha=0.7), ha='left', va='bottom')
#         self.tooltip.set_visible(False)
        
#         # 绑定鼠标移动事件
#         self.fig.canvas.mpl_connect('motion_notify_event', self._on_mouse_move)
    
#     def _on_mouse_move(self, event):
#         if event.inaxes != self.ax:
#             self.tooltip.set_visible(False)
#             self.fig.canvas.draw_idle()
#             return
        
#         x, y = event.xdata, event.ydata
#         info = f"x={x:.2f}, y={y:.2f}"  # 构造提示信息
#         self.tooltip.set_text(info)
#         self.tooltip.set_position((x, y))
#         self.tooltip.set_visible(True)
#         self.fig.canvas.draw_idle()

# # 创建实例并显示图表
# plot = PlotWithTooltip()
# plt.show()


from matplotlib.patches import Arrow

fig, ax = plt.subplots(figsize=(6, 4))

from matplotlib.patches import FancyArrow

arrow = FancyArrow(
    x=0, y=0,          # 起点
    dx=0.3, dy=0.3,        # 方向向量（决定长度）
    width=0.03,         # 尾部宽度
    head_width=0.08,    # 头部宽度
    head_length=0.06,   # 头部长度
    color='red',
    alpha=0.8
)
ax.add_patch(arrow)

ax.set_xlim(0, 8)
ax.set_ylim(0, 3)
plt.show()