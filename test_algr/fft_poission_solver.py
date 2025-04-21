import numpy as np
from numpy.fft import fft2, ifft2
import time

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

# 有限差分法（雅可比迭代）
def solve_poisson_fdm(rho, region, tol=1e-6, max_iter=10000):
    """
    使用有限差分法（雅可比迭代）求解二维泊松方程：nabla^2 psi = -rho
    rho: 密度矩阵 (m x m)
    dx: 网格间距
    tol: 收敛容差
    max_iter: 最大迭代次数
    返回：电势 psi
    """
    m, n = rho.shape
    psi = np.zeros((m, n))  # 初始化电势为 0
    dx = region[0] / m
    dy = region[1] / n

    rho_dx2 = rho * dx**2  # 预乘 dx^2 以优化计算
    
    
    for iteration in range(max_iter):
        psi_old = psi.copy()
        # 更新内部网格点（边界保持 psi = 0）
        for i in range(1, m-1):
            for j in range(1, n-1):
                psi[i, j] = (psi_old[i+1, j] + psi_old[i-1, j] + 
                            psi_old[i, j+1] + psi_old[i, j-1] + rho_dx2[i, j]) / 4.0
        # 检查收敛
        residual = np.max(np.abs(psi - psi_old))
        if residual < tol:
            print(f"FDM 收敛于 {iteration+1} 次迭代，残差 = {residual:.2e}")
            break
    else:
        print(f"FDM 未在 {max_iter} 次迭代内收敛，残差 = {residual:.2e}")
    
    return psi

# 计算残差以评估精度
def compute_residual(psi, rho, dx):
    """
    计算泊松方程的残差：nabla^2 psi + rho
    残差越小，解越精确
    """
    m, n = psi.shape
    residual = np.zeros((m, n))
    for i in range(1, m-1):
        for j in range(1, n-1):
            laplacian = (psi[i+1, j] + psi[i-1, j] + psi[i, j+1] + psi[i, j-1] - 4*psi[i, j]) / dx**2
            residual[i, j] = laplacian + rho[i, j]
    return np.max(np.abs(residual))

# 主程序：比较 FFT 和 FDM
def compare_methods(m=1024, L=10240.0):
    """
    比较 FFT 和 FDM 的时间和精度
    m: 网格点数
    L: 区域边长
    """
    dx = L / m  # 网格间距
    rho = np.random.rand(m, m)  # 随机密度
    
    # FFT 方法
    start_time = time.time()
    psi_fft = solve_poisson_fft(rho, dx=dx)

    fft_time = time.time() - start_time
    fft_residual = compute_residual(psi_fft, rho, dx)
    
    # FDM 方法
    start_time = time.time()
    psi_fdm = solve_poisson_fdm(rho, dx=dx, tol=1e-6, max_iter=10000)
    fdm_time = time.time() - start_time
    fdm_residual = compute_residual(psi_fdm, rho, dx)
    
    # 解之间的差异
    diff_psi = np.max(np.abs(psi_fft - psi_fdm))
    
    # 输出结果
    print(f"\n比较结（网格 {m}x{m}, 区域 {L}x{L}, dx = {dx}):")
    print(f"FFT 方法:")
    print(f"  时间: {fft_time:.4f} 秒")
    print(f"  残差: {fft_residual:.2e}")
    print(f"FDM 方法:")
    print(f"  时间: {fdm_time:.4f} 秒")
    print(f"  残差: {fdm_residual:.2e}")
    print(f"两种解的最大差异: {diff_psi:.2e}")

# 运行比较
if __name__ == "__main__":
    compare_methods(m=16,L=1024.0)
    compare_methods(m=32, L=1024.0)
    compare_methods(m=64, L=1024.0)
    compare_methods(m=128, L=1024.0)
