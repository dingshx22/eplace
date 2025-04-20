import numpy as np

# 构造简单的正定二次函数
np.random.seed(0)
n = 128
A = np.random.rand(n, n)
A = A.T @ A + np.eye(n)  # 保证正定
b = np.random.rand(n)

# 理论最优解
x_true = -np.linalg.inv(A) @ b

# 初始点
x0 = np.zeros(n)
max_iter = 100

def backtracking_line_search(x, g, f, alpha_init=1.0, rho=0.3, c=0.3, max_iter=10):
    alpha = alpha_init
    for _ in range(max_iter):
        x_new = x - alpha * g
        if f(x_new) <= f(x) - c * alpha * np.dot(g, g):
            return alpha
        alpha *= rho
    return alpha



# 函数和梯度
def f(x): return 0.5 * x @ A @ x + b @ x
def grad(x): return A @ x + b

# -------------------
# 方法 1：标准牛顿法
# -------------------
x_newton = x0.copy()
for _ in range(max_iter):
    g = grad(x_newton)
    H = A  # 精确 Hessian
    delta = np.linalg.solve(H, g)
    x_newton -= delta

# -----------------------------
# 方法 2：对角预处理器近似牛顿法
# -----------------------------
# 修复后的预处理近似牛顿法

lambda_q = 10
alpha_init = 1  # 控制步长
x_diag = x0.copy()

for _ in range(max_iter):
    g = grad(x_diag)
    P_diag = 1.0 / (np.diag(A) + lambda_q)
    alpha = backtracking_line_search(x_diag, g, f, alpha_init=alpha_init)
    delta = P_diag * g
    x_diag -= alpha * delta


# -------------------
# 打印结果对比
# -------------------
print("理论最优解:", x_true,x_true.shape)
print("牛顿法结果: ", x_newton,x_newton.shape)
print("对角预处理结果:", x_diag,x_diag.shape)

print("\n误差对比：")
print("牛顿法误差: ", np.linalg.norm(x_newton - x_true))
print("预处理误差: ", np.linalg.norm(x_diag - x_true))
