import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dst, idst
from src.utility import calculate_field, solve_poisson_dst


m, n = 64, 64
width, height = 1.0, 1.0
x = np.linspace(0, width, m)
y = np.linspace(0, height, n)
X, Y = np.meshgrid(x, y)
potential = X**2 + Y**2  # 解析电势



bin_width = width / n
bin_height = height / m



# 计算电场
field_x = np.zeros_like(potential)
field_y = np.zeros_like(potential)
field_x[1:-1, :] = -(potential[2:, :] - potential[:-2, :]) / (2 * bin_height)
field_x[0, :] = -(potential[1, :] - potential[0, :]) / bin_height
field_x[-1, :] = -(potential[-1, :] - potential[-2, :]) / bin_height
field_y[:, 1:-1] = -(potential[:, 2:] - potential[:, :-2]) / (2 * bin_width)
field_y[:, 0] = -(potential[:, 1] - potential[:, 0]) / bin_width
field_y[:, -1] = -(potential[:, -1] - potential[:, -2]) / bin_width

# 解析解
field_x_analytic = -2 * X
field_y_analytic = -2 * Y

# 计算误差
error_x = np.abs(field_x - field_x_analytic)
error_y = np.abs(field_y - field_y_analytic)

print("Max error in field_x:", np.max(error_x))
print("Max error in field_y:", np.max(error_y))

# 可视化误差
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(error_x, cmap='viridis')
plt.title("Error in Field X")
plt.colorbar()
plt.subplot(1, 2, 2)
plt.imshow(error_y, cmap='viridis')
plt.title("Error in Field Y")
plt.colorbar()
plt.show()