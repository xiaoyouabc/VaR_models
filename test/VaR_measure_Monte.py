# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from Data_loader import Dataloader
from VaR.calculate_functions import calculate_returns, calculate_volatility_mean
import warnings
warnings.filterwarnings('ignore')


def monte_carlo_returns_basic(mu, sigma, T, steps, simulations):
    """
    基于正态分布的蒙特卡洛模拟生成收益率路径
    :param mu: 预期收益率
    :param sigma: 波动率
    :param T: 时间（年）
    :param steps: 每年步数
    :param simulations: 模拟次数
    :return: 模拟的收益率路径
    """
    dt = T / steps
    returns = np.random.normal(mu * dt, sigma * np.sqrt(dt), (steps, simulations))
    return returns


def monte_carlo_returns_gbm(mu, sigma, T, steps, simulations):
    """
    基于几何布朗运动（GBM）的收益率路径模拟
    """
    dt = T / steps
    returns = np.random.normal((mu - 0.5 * sigma ** 2) * dt, sigma * np.sqrt(dt), (steps, simulations))
    return returns


'''
def monte_carlo_returns_heston(mu, sigma0, kappa, theta, xi, rho, T, steps, simulations):
    """
    Heston 模型的蒙特卡洛模拟生成收益率路径
    """
    dt = T / steps
    vol_paths = np.zeros((steps, simulations))
    returns = np.zeros((steps, simulations))
    vol_paths[0] = sigma0

    for t in range(1, steps):
        z1 = np.random.standard_normal(simulations)
        z2 = np.random.standard_normal(simulations)
        z2 = rho * z1 + np.sqrt(1 - rho ** 2) * z2

        vol_paths[t] = vol_paths[t - 1] + kappa * (theta - vol_paths[t - 1]) * dt + xi * np.sqrt(
            np.abs(vol_paths[t - 1])) * np.sqrt(dt) * z2

        vol_paths[t] = np.maximum(vol_paths[t], 0)  # 保证波动率为正

        returns[t] = np.random.normal(mu * dt, np.sqrt(vol_paths[t] * dt), simulations)

    return returns
'''


def monte_carlo_returns_jump_diffusion(mu, sigma, lambd, jump_mu, jump_sigma, T, steps, simulations):
    """
    跳跃扩散模型的收益率路径模拟
    """
    dt = T / steps
    returns = np.zeros((steps, simulations))

    for t in range(steps):
        z = np.random.standard_normal(simulations)
        jumps = np.random.poisson(lambd * dt, simulations)
        jump_size = np.random.normal(jump_mu, jump_sigma, simulations) * jumps

        returns[t] = mu * dt + sigma * np.sqrt(dt) * z + jump_size

    return returns


def calculate_var(returns, confidence_level=0.95):
    """
    计算每个时间点的 VaR 值
    """
    var_series = np.percentile(returns, (1 - confidence_level) * 100, axis=1)
    return var_series


def plot_returns_and_var(historical_returns, var_series, title):
    """
    绘制历史收益率和未来 VaR，并添加中位数虚线。
    :param historical_returns: 历史收益率序列
    :param var_series: 未来 VaR 序列
    :param title: 图的标题
    """
    # 计算 VaR 四分位数
    q25 = np.percentile(var_series, 25)

    # 统计超过中位数的 VaR 值数量
    count_above_25 = np.sum(historical_returns < q25)
    print(f'Number of VaR values below the 5-percent: {count_above_25}')

    plt.figure(figsize=(12, 6))

    # 绘制历史收益率（蓝色）
    plt.plot(historical_returns, label='Historical Returns', color='blue')

    # 绘制未来 VaR（红色）
    steps = len(var_series)
    plt.plot(range(len(historical_returns), len(historical_returns) + steps),
             var_series, label='Forecasted VaR', color='red')

    # 添加中位数虚线（紫色）
    plt.axhline(y=q25, color='purple', linestyle='--', label=' VaR (25% Quantile)')

    plt.title(title)
    plt.xlabel('Time Steps')
    plt.ylabel('Returns / VaR')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()


if __name__ == '__main__':

    data = Dataloader()
    data = calculate_returns(data).reset_index(drop=True)
    mu, sigma = calculate_volatility_mean(data)
    # 参数设置
    # sigma0 = 0.2  # 初始波动率（Heston 模型）
    # kappa = 2.0  # 回归速度
    theta = 0.2  # 波动率长期均值
    xi = 0.1  # 波动率波动幅度
    rho = -0.5  # 相关系数
    lambd = 0.75  # 跳跃强度
    jump_mu = 0.02  # 跳跃幅度均值
    jump_sigma = 0.1  # 跳跃幅度波动率
    T = 1  # 模拟1年
    steps = 252  # 每年252个交易日
    simulations = 10000  # 模拟10000次
    confidence_level = 0.95  # VaR置信水平

    # 模拟收益率路径
    returns_basic = monte_carlo_returns_basic(mu, sigma, T, steps, simulations)
    returns_gbm = monte_carlo_returns_gbm(mu, sigma, T, steps, simulations)
    # returns_heston = monte_carlo_returns_heston(mu, sigma0, kappa, theta, xi, rho, T, steps, simulations)
    returns_jump_diffusion = monte_carlo_returns_jump_diffusion(mu, sigma, lambd, jump_mu, jump_sigma, T, steps, simulations)

    # 计算 VaR
    var_basic = calculate_var(returns_basic, confidence_level)
    var_gbm = calculate_var(returns_gbm, confidence_level)
    # var_heston = calculate_var(returns_heston, confidence_level)
    var_jump_diffusion = calculate_var(returns_jump_diffusion, confidence_level)

    # 绘制 VaR 曲线
    plot_returns_and_var(data, var_basic, 'VaR - Basic Monte Carlo Simulation')
    plot_returns_and_var(data, var_gbm, 'VaR - GBM Simulation')
    # plot_var(var_heston, 'VaR - Heston Model Simulation')
    plot_returns_and_var(data, var_jump_diffusion, 'VaR - Jump Diffusion Model Simulation')
