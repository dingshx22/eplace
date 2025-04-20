import numpy as np
import time


# 目标函数及其梯度
def objective_function(x, A, b):
    return 0.5 * np.dot(x, A @ x) + np.dot(b, x)

def gradient(x, A, b):
    return A @ x + b


# 预处理器（增强正则化）
def preconditioner(A, lambda_q):
    diag = np.diag(A) + lambda_q  # 增加正则化项
    return 1.0 / diag


# 回溯线搜索（记录尝试次数并检查数值稳定性）
def backtracking_line_search(x, g, f, alpha_init=0.1, rho=0.5, c=0.05, max_iter=10):
    alpha = alpha_init
    attempts = 0
    start_time = time.time()
    for _ in range(max_iter):
        x_new = x - alpha * g
        f_new = f(x_new)
        if np.isinf(f_new) or np.isnan(f_new):
            alpha *= rho
            attempts += 1
            continue
        if f_new <= f(x) - c * alpha * np.dot(g, g):
            return alpha, attempts + 1, time.time() - start_time
        alpha *= rho
        attempts += 1
    return alpha, attempts + 1, time.time() - start_time


# Nesterov 加速梯度下降（修复与优化版）
def nesterov_gradient_descent(A, b, x0, lambda_q=10.0, max_iter=100, tol=1e-5):
    x = x0.copy()
    y = x0.copy()
    v = x0.copy()
    t = 1.0
    alpha_init = 0.1  # 保守的初始步长
    
    # 记录时间
    timing = {
        'lookahead': 0.0,
        'gradient': 0.0,
        'preconditioner': 0.0,
        'line_search': 0.0,
        'update': 0.0,
        'momentum': 0.0,
        'convergence': 0.0
    }
    total_iterations = 0
    total_line_search_attempts = 0
    
    for k in range(max_iter):
        # 前瞻点计算
        start_time = time.time()
        y = x + ((t - 1) / (t + 1)) * (x - v)
        timing['lookahead'] += time.time() - start_time
        
        # 梯度计算
        start_time = time.time()
        grad = gradient(y, A, b)
        timing['gradient'] += time.time() - start_time
        
        # 检查梯度是否有效
        if np.any(np.isinf(grad)) or np.any(np.isnan(grad)):
            print(f"Iteration {k+1}: Gradient overflow or NaN detected.")
            break
        
        # 预处理器
        start_time = time.time()
        P = preconditioner(A, lambda_q)
        timing['preconditioner'] += time.time() - start_time
        
        # 预处理梯度
        start_time = time.time()
        g = P * grad
        timing['update'] += time.time() - start_time
        
        # 检查预处理梯度是否有效
        if np.any(np.isinf(g)) or np.any(np.isnan(g)):
            print(f"Iteration {k+1}: Preconditioned gradient overflow or NaN detected.")
            break
        
        # 回溯线搜索
        alpha, attempts, ls_time = backtracking_line_search(
            y, g, lambda x: objective_function(x, A, b),
            alpha_init=alpha_init, rho=0.5, c=0.05, max_iter=10
        )
        timing['line_search'] += ls_time
        total_line_search_attempts += attempts
        
        # 更新
        start_time = time.time()
        v = x
        x = y - alpha * g
        timing['update'] += time.time() - start_time
        
        # 检查更新后的点是否有效
        if np.any(np.isinf(x)) or np.any(np.isnan(x)):
            print(f"Iteration {k+1}: Update resulted in overflow or NaN.")
            break
        
        # 动量参数更新（修复 ^ 为 **）
        start_time = time.time()
        t_new = (1 + np.sqrt(1 + 4 * t**2)) / 2
        t = t_new
        timing['momentum'] += time.time() - start_time
        
        # 收敛检查
        start_time = time.time()
        grad_norm = np.linalg.norm(grad)
        timing['convergence'] += time.time() - start_time
        
        total_iterations += 1
        
        # 打印每次迭代的调试信息（可选）
        print(f"Iteration {k+1}: Gradient Norm = {grad_norm:.6e}, Line Search Attempts = {attempts}")
        
        if grad_norm < tol:
            print(f"Converged at iteration {k+1} with gradient norm {grad_norm:.6e}")
            break
    
    # 打印总结
    print(f"\nNesterov 总迭代次数: {total_iterations}")
    print(f"平均每次迭代的线搜索尝试次数: {total_line_search_attempts / total_iterations:.2f}")
    print("各步骤总耗时 (秒):")
    for step, time_spent in timing.items():
        print(f"  {step}: {time_spent:.6f} s ({time_spent / sum(timing.values()) * 100:.2f}%)")
    
    return x


# 示例：优化二次函数
np.random.seed(0)
n = 4096
A = np.random.rand(n, n)
A = A @ A.T + 10.0 * np.eye(n)  # 增加正则化，降低条件数
b = np.random.rand(n)
x0 = np.zeros(n)
lambda_q = 10.0  # 增强预处理器正则化

# 计算理论最优解
start_time = time.time()
x_true = -np.linalg.inv(A) @ b
end_time = time.time()
print("理论最优解计算时间:", end_time - start_time, "seconds")

# 运行 Nesterov 方法
start_time = time.time()
x_opt = nesterov_gradient_descent(A, b, x0, lambda_q)
end_time = time.time()
print("Nesterov 总计算时间:", end_time - start_time, "seconds")

# 结果比较
print("Nesterov 结果的 Objective value:", objective_function(x_opt, A, b))
print("理论最优解的 Objective value:", objective_function(x_true, A, b))
print("Nesterov 结果与理论最优解的误差:", np.linalg.norm(x_opt - x_true))