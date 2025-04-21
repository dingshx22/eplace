import numpy as np
from scipy.fftpack import dst, idst

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
    rho_smooth = rho

    m, n = rho_smooth.shape
    dx = region[0] / m
    dy = region[1] / n
    
    # Create wave numbers for DST (Dirichlet BCs)
    i = np.arange(1, m + 1)  # 1 to m (1 to 64)
    j = np.arange(1, n + 1)  # 1 to n (1 to 64)
    kx = np.pi * i / region[0]
    ky = np.pi * j / region[1]
    kx, ky = np.meshgrid(kx, ky)
    
    denom = kx**2 + ky**2
    epsilon = 1e-10
    denom = np.where(denom == 0, epsilon, denom)

    # Use norm="ortho" for correct scaling
    rho_dst = dst(dst(rho_smooth, type=2, axis=0, norm="ortho"), type=2, axis=1, norm="ortho")
    psi_dst = -rho_dst / denom
    psi = idst(idst(psi_dst, type=2, axis=1, norm="ortho"), type=2, axis=0, norm="ortho")
    
    # Check for numerical issues
    if np.any(np.isnan(psi)) or np.any(np.isinf(psi)):
        print("Numerical instability detected in Poisson solver")
        psi = np.zeros_like(rho)
    
    return psi

import numpy as np
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