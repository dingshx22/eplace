import numpy as np
import matplotlib.pyplot as plt

def f(v):
    x, y = v
    return (x - 2)**2 + 2*(y + 3)**2 + np.sin(3*x) * np.sin(4*y)

def grad_f(v):
    x, y = v
    dfdx = 2*(x - 2) + 3*np.cos(3*x)*np.sin(4*y)
    dfdy = 4*(y + 3) + 4*np.sin(3*x)*np.cos(4*y)
    return np.array([dfdx, dfdy])

def nesterov_lipschitz(f, grad, x0, mu=0.9, iterations=100):
    x = x0.copy()
    v = np.zeros_like(x)
    history = [x.copy()]
    g_prev = grad(x)
    x_prev = x.copy()

    for i in range(iterations):
        y = x + mu * v
        g = grad(y)

        # Lipschitz 估计（从第二步开始）
        if i > 0:
            diff_g = g - g_prev
            diff_x = y - x_prev
            L = np.linalg.norm(diff_g) / (np.linalg.norm(diff_x) + 1e-8)  # 避免除0
        else:
            L = 10.0  # 初始猜测

        eta = 1.0 / L  # 步长 = 1/L
        v = mu * v - eta * g
        x = x + v

        # 保存状态
        history.append(x.copy())
        g_prev = g
        x_prev = y

    return np.array(history)

# 运行
nag_lip_path = nesterov_lipschitz(f, grad_f, x0=np.array([0.0, 0.0]))

# 画图
xlist = np.linspace(-2, 5, 400)
ylist = np.linspace(-6, 1, 400)
X, Y = np.meshgrid(xlist, ylist)
Z = f([X, Y])

plt.figure(figsize=(10, 6))
plt.contour(X, Y, Z, levels=50, cmap='jet')
plt.plot(nag_lip_path[:, 0], nag_lip_path[:, 1], 'o-', label='Nesterov + Lipschitz')
plt.title("Nesterov with Lipschitz Step Size")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()
