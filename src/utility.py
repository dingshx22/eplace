import numpy as np
from numpy.fft import fft2, ifft2
import time
from scipy.ndimage import gaussian_filter
from scipy.fft import dst, idst
from typing import Tuple, Callable

# FFT 方法
def solve_poisson_fft(rho, region):
    """
    使用 FFT 求解二维泊松方程:nabla^2 psi = -rho
    rho: 密度矩阵 (m x m)
    dx: 网格间距
    返回：电势 psi
    """
    m, n = rho.shape
    dx = region[0] / m
    kx = np.fft.fftfreq(m, d=dx) * 2 * np.pi
    dx= region[1]/n
    ky = np.fft.fftfreq(n, d=dx) * 2 * np.pi
    kx, ky = np.meshgrid(kx, ky)
    denom = kx**2 + ky**2
    denom[0, 0] = 1.0  # 避免除零
    rho_hat = fft2(rho)
    psi_hat = rho_hat / denom
    psi_hat[0, 0] = 0.0  # 移除直流分量
    psi = np.real(ifft2(psi_hat))

    return psi

def solve_poisson_dst(rho, region, sigma=1.0):
    """
    Solve 2D Poisson equation (∇²ψ = -ρ) using Discrete Sine Transform (DST)
    with Dirichlet boundary conditions (ψ = 0 at edges).
    Args:
        rho: Density matrix (m x n)
        region: Tuple of (width, height) of the physical region
        sigma: Standard deviation for Gaussian smoothing
    """
    rho_smooth = gaussian_filter(rho, sigma=sigma)    # Apply Gaussian smoothing to density
    # rho_smooth = rho

    rho_mean = np.mean(rho_smooth)
    rho_smooth = rho_smooth - rho_mean

    m, n = rho_smooth.shape
    dx = region[0] / m
    dy = region[1] / n

    # Create wave numbers for DST (Dirichlet BCs)
    i = np.arange(1, m + 1)
    j = np.arange(1, n + 1)
    kx = (np.pi / (m * dx)) * i
    ky = (np.pi / (n * dy)) * j
    kx, ky = np.meshgrid(kx, ky)

    denom = kx**2 + ky**2    # Compute denominator (k^2 = kx^2 + ky^2)

    epsilon = 1e-10    # Avoid division by zero with small epsilon
    denom = np.where(denom == 0, epsilon, denom)

    # Use norm="ortho" for correct scaling
    rho_dst = dst(dst(rho_smooth, type=2, axis=0,norm='ortho'), type=2, axis=1,norm='ortho')
    psi_dst = rho_dst / denom    # Solve in frequency domain: ψ̂ = -ρ̂ / (kx^2 + ky^2)  ，
    # 但是在代码中不加负号反而正确不知道为什么

    # Inverse DST to get potential
    psi = idst(idst(psi_dst, type=2, axis=1,norm='ortho'), type=2, axis=0,norm='ortho')

    # Normalize to account for DST scaling
    # psi /= (2 * (m + 1)) * (2 * (n + 1))
    # psi /= (4 * m * n)  # 尝试不同的归一化因子

    psi_mean = np.mean(psi)
    psi = psi - psi_mean

    # Check for numerical issues
    if np.any(np.isnan(psi)) or np.any(np.isinf(psi)):
        print("Numerical instability detected in Poisson solver")
        psi = np.zeros_like(rho)

    return psi

def calculate_field(potential, region) :
    """
    计算电场
    """
    field_x = np.zeros_like(potential)
    field_y = np.zeros_like(potential)

    m,n=potential.shape
    bin_width=region[0] / m
    bin_height=region[1] / n

    # 计算x方向的电场   中心区域使用中心差分
    field_x[1:-1, :] = -(potential[2:, :] - potential[:-2, :]) / (2 * bin_height)
    # 边界使用单侧差分
    field_x[0, :] = -(potential[1, :] - potential[0, :]) / bin_height
    field_x[-1, :] = -(potential[-1, :] - potential[-2, :]) / bin_height

    # 计算y方向的电场  中心区域使用中心差分
    field_y[:, 1:-1] = -(potential[:, 2:] - potential[:, :-2]) / (2 * bin_width)
    # 边界使用单侧差分
    field_y[:, 0] = -(potential[:, 1] - potential[:, 0]) / bin_width
    field_y[:, -1] = -(potential[:, -1] - potential[:, -2]) / bin_width

    return field_x, field_y

class NesterovOptimizer:
    def __init__(self,
                 f: Callable[[np.ndarray], float],               # 目标函数
                 grad_f: Callable[[np.ndarray], np.ndarray],     # 目标函数的梯度
                 L: float,                                       # Lipschitz常数
                 x0: np.ndarray,                                 # 初始点
                 mu: float = 0.9,                                # 动量参数
                 max_iter: int = 1000,                           # 最大迭代次数
                 tol: float = 1e-6):                             # 收敛容差
        self.f = f
        self.grad_f = grad_f
        self.L = L
        self.x0 = x0
        self.mu = mu
        self.max_iter = max_iter
        self.tol = tol


    def optimize(self) -> Tuple[np.ndarray, list]:
        x = self.x0.copy()
        v = np.zeros_like(x)  # 动量项
        history = []

        for _ in range(self.max_iter):
            y = x + self.mu * v                     # 计算预测点
            grad = self.grad_f(y)                   # 计算梯度
            v = self.mu * v - (1/self.L) * grad     # 更新动量
            x_new = x + v                           # 更新位置
            history.append(self.f(x_new))
            if np.linalg.norm(x_new - x) < self.tol:
                break
            x = x_new

        return x, history

def estimate_lipschitz_constant(grad_f: Callable[[np.ndarray], np.ndarray], 
                                x0: np.ndarray,                              # 初始点                                    
                                num_samples: int = 100) -> float:            # 采样点数
    L = 0.0
    x = x0.copy()
    print(f"x.shape: {x.shape}")

    for _ in range(num_samples):
        # 生成随机方向
        direction = np.random.randn(*x.shape)
        print(f"direction.norm: {np.linalg.norm(direction)}")
        print(f"direction.shape: {direction.shape}")
        direction = direction / np.linalg.norm(direction)

        # 计算梯度差
        grad1 = grad_f(x)
        print(f"grad1.norm: {np.linalg.norm(grad1)}")

        grad2 = grad_f(x + 1e-4 * direction)
        print(f"grad2.norm: {np.linalg.norm(grad2)}")

        grad_diff = np.linalg.norm(grad2 - grad1)
        print(f"grad_diff: {grad_diff}")


        L = max(L, grad_diff / 1e-4)  # 更新Lipschitz常数估计
    return L




if __name__ == "__main__":
    import matplotlib.pyplot as plt
    m, n = 64, 64
    rho = np.zeros((m, n))
    rho[m//2, n//2] = 1.0

    # psi = solve_poisson_dst(rho, region=(1.0, 1.0))
    psi2=solve_poisson_fft(rho, region=(1.0, 1.0))

    # print(psi.sum())
    print(psi2.sum())

    # plt.imshow(psi, cmap='viridis', vmin=None, vmax=None)  # 自动调整色条范围
    plt.imshow(psi2, cmap='viridis', vmin=None, vmax=None)  # 自动调整色条范围
    plt.title("Potential of Point Charge")

    plt.colorbar()
    plt.show()


    # rho = np.zeros((64, 64))
    # rho[30, 32] = 1.0
    # rho[34, 32] = -1.0
    # psi = solve_poisson_dst(rho, region=(1.0, 1.0))

    # plt.imshow(psi, cmap='viridis', vmin=None, vmax=None)  # 自动调整色条范围
    # plt.title("Potential of Two Charges")
    # plt.colorbar()
    # plt.show()
