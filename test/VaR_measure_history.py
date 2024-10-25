# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings


warnings.filterwarnings('ignore')





def plot_returns_and_var(prices, var_series, title='Returns and VaR'):
    """
    绘制收益率和 VaR 的图表，使用单一 y 轴
    :param prices: 资产价格的时间序列数据
    :param var_series: VaR 的序列
    :param title: 图表标题
    """
    # 计算对数收益率
    returns = np.log(prices / prices.shift(1)).dropna()
    plt.figure(figsize=(10, 6))
    # 绘制收益率
    plt.plot(returns.index, returns, label='Returns', color='blue')
    # 绘制 VaR
    plt.plot(var_series.index, var_series, label='VaR', color='red', linestyle='--')
    plt.fill_between(var_series.index, var_series, alpha=0.2, color='red')
    # 添加标题和标签
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Returns / VaR')
    # 添加图例
    plt.legend(loc='best')
    # 显示网格
    plt.grid(True)
    # 显示图表
    plt.show()


def plot_returns_var_and_cvar(prices, var_series, cvar_series, title='Returns, VaR, and CVaR'):
    """
    绘制收益率、VaR 和 CVaR 的图表，使用单一 y 轴
    :param prices: 资产价格的时间序列数据
    :param var_series: VaR 的序列
    :param cvar_series: CVaR 的序列
    :param title: 图表标题
    """
    returns = np.log(prices / prices.shift(1)).dropna()  # 计算对数收益率
    plt.figure(figsize=(10, 6))
    # 绘制收益率
    plt.plot(returns.index, returns, label='Returns', color='blue')
    # 绘制 VaR
    plt.plot(var_series.index, var_series, label='VaR', color='red', linestyle='--')
    # 绘制 CVaR
    plt.plot(cvar_series.index, cvar_series, label='CVaR', color='green', linestyle='-.')
    # 填充 VaR 下方区域
    plt.fill_between(var_series.index, var_series, alpha=0.2, color='red')
    # 填充 CVaR 下方区域
    plt.fill_between(cvar_series.index, cvar_series, alpha=0.1, color='green')
    # 添加标题和标签
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Returns / VaR / CVaR')
    # 添加图例
    plt.legend(loc='best')
    # 显示网格
    plt.grid(True)
    # 显示图表
    plt.show()


if __name__ == '__main__':

    # 示例价格数据
    np.random.seed(42)
    prices = pd.Series(np.random.normal(100, 1, 1000), name='price')

    # 设置回溯窗口和置信水平
    window = 100  # 回溯窗口期（如100天）
    confidence_level = 0.95  # VaR 置信水平
    # 计算历史模拟 VaR
    var_series = calculate_historical_VaR(prices, window, confidence_level)
    # 绘制价格和 VaR
    # 绘制收益率和 VaR
    plot_returns_and_var(prices[100:], var_series, title=f'VaR with Window={window} and Confidence={confidence_level}')

    cvar_series = calculate_CVaR_history(prices, window, confidence_level)
    # 绘制收益率、VaR 和 CVaR
    plot_returns_var_and_cvar(prices[100:], var_series, cvar_series, title=f'VaR and CVaR with Window={window} '
                                                                           f'and Confidence={confidence_level}')


