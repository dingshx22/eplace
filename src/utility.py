import numpy as np
from numpy.fft import fft2, ifft2
import time
from scipy.ndimage import gaussian_filter
from scipy.fft import dst, idst


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
    
    Returns:
        psi: Potential matrix (m x n)
    """
    # Apply Gaussian smoothing to density
    rho_smooth = gaussian_filter(rho, sigma=sigma)

    # Remove DC component from density (make zero-mean)
    rho_mean = np.mean(rho_smooth)
    rho_smooth = rho_smooth - rho_mean
    # print(f"Removed DC component from density: mean = {rho_mean:.4e}")


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
    rho_dst = dst(dst(rho_smooth, type=2, axis=0), type=2, axis=1)    # Apply DST to rho
    psi_dst = -rho_dst / denom    # Solve in frequency domain: ψ̂ = -ρ̂ / (kx^2 + ky^2)
    
    # Inverse DST to get potential
    psi = idst(idst(psi_dst, type=2, axis=1), type=2, axis=0)
    
    # Normalize to account for DST scaling
    psi /= (2 * (m + 1)) * (2 * (n + 1))

    # Remove DC component from potential (ensure zero-mean)
    psi_mean = np.mean(psi)
    psi = psi - psi_mean
    # print(f"Removed DC component from potential: mean = {psi_mean:.4e}")
    
    # Check for numerical issues
    if np.any(np.isnan(psi)) or np.any(np.isinf(psi)):
        print("Numerical instability detected in Poisson solver")
        psi = np.zeros_like(rho)
    
    return psi
