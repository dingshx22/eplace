import numpy as np
from scipy.fftpack import dst, idst
from scipy.ndimage import gaussian_filter


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

    return psi



from scipy.fftpack import dst, idst
import matplotlib.pyplot as plt

def solve_poisson_dst(rho, region, sigma=1.0):
    rho_smooth = rho

    m, n = rho_smooth.shape
    dx = region[0] / m
    dy = region[1] / n
    
    i = np.arange(1, m + 1)
    j = np.arange(1, n + 1)
    kx = np.pi * i / region[0]
    ky = np.pi * j / region[1]
    kx, ky = np.meshgrid(kx, ky)
    
    denom = kx**2 + ky**2
    epsilon = 1e-10
    denom = np.where(denom == 0, epsilon, denom)

    rho_dst = dst(dst(rho_smooth, type=2, axis=0, norm="ortho"), type=2, axis=1, norm="ortho")
    psi_dst = rho_dst / denom
    psi = idst(idst(psi_dst, type=2, axis=1, norm="ortho"), type=2, axis=0, norm="ortho")
    
    if np.any(np.isnan(psi)) or np.any(np.isinf(psi)):
        print("Numerical instability detected in Poisson solver")
        psi = np.zeros_like(rho)
    
    return psi

# Test script
m, n = 64, 64
rho = np.zeros((m, n))
rho[m//2, n//2] = 1.0

psi = solve_poisson_dst(rho, region=(1.0, 1.0))

plt.imshow(psi, cmap='viridis')
plt.title("Potential of Point Charge")
plt.colorbar()
plt.show()