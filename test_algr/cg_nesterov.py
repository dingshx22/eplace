import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# 定义目标函数
def f(v):
    x, y = v
    return (x - 2)**2 + 2*(y + 3)**2 + np.sin(3*x) * np.sin(4*y)

# 梯度（用于 Nesterov）
def grad_f(v):
    x, y = v
    dfdx = 2*(x - 2) + 3*np.cos(3*x)*np.sin(4*y)
    dfdy = 4*(y + 3) + 4*np.sin(3*x)*np.cos(4*y)
    return np.array([dfdx, dfdy])

# ===== 共轭梯度法 CG（使用scipy封装） =====
res_cg = minimize(f, x0=np.array([0.0, 0.0]), method='CG', options={"return_all": True})
cg_path = np.array(res_cg.allvecs)

# ===== Nesterov Accelerated Gradient 手动实现 =====
def nesterov(f, grad, x0, lr=0.01, mu=0.9, iterations=100):
    x = x0.copy()
    v = np.zeros_like(x)
    history = [x.copy()]
    for _ in range(iterations):
        lookahead = x + mu * v
        g = grad(lookahead)
        v = mu * v - lr * g
        x += v
        history.append(x.copy())
    return np.array(history)

nag_path = nesterov(f, grad_f, np.array([0.0, 0.0]))

# ====== 画出路径对比 ======
xlist = np.linspace(-2, 5, 400)
ylist = np.linspace(-6, 1, 400)
X, Y = np.meshgrid(xlist, ylist)
Z = f([X, Y])

plt.figure(figsize=(10, 6))
plt.contour(X, Y, Z, levels=50, cmap='jet')
plt.plot(cg_path[:, 0], cg_path[:, 1], 'o-', label='Conjugate Gradient')
plt.plot(nag_path[:, 0], nag_path[:, 1], 's-', label='Nesterov AG')
plt.legend()
plt.title("Optimization Path Comparison")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.show()
