# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def calculate_and_visualize_coverage(Returns, VaR):
    # 计算投资组合价值的损失
    losses = np.abs(Returns)
    # 确定损失是否小于或等于VaR值
    coverage = (losses <= VaR)
    # 计算所涵盖结果的比例
    coverage_ratio = np.mean(coverage)
    # 可视化
    plt.figure(figsize=(40, 15))
    plt.plot(Returns, label='收益率序列')
    plt.axhline(y=np.mean(VaR), color='r', linestyle='--', label='平均VaR')
    plt.fill_between(range(len(Returns)), VaR, color='g', alpha=0.3, label='VaR覆盖区域')
    plt.scatter(range(len(Returns)), losses, color='b', label='损失')
    plt.title('收益率序列与VaR覆盖')
    plt.xlabel('时间点')
    plt.ylabel('收益率/损失')
    plt.legend()
    plt.show()

    return coverage_ratio


def plot_VaR_and_returns_0(returns, VaR_series, title='Returns and VaR (GARCH + GED)', xlabel='Time', ylabel='VaR', output_dir=None):
    """
    绘制收益率和VaR曲线在同一y轴上
    :param returns: 收益率序列
    :param title: 图片标题
    :param VaR_series: VaR序列
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    # 绘制收益率
    ax.plot(returns.index, returns, color='blue', label='Returns', alpha=0.6)
    # 绘制VaR曲线
    ax.plot(VaR_series.index, VaR_series, color='red', label='VaRs', alpha=0.6)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # 设置图例
    ax.legend(loc='best')
    # 标题
    plt.title(title)
    # 调整布局
    fig.tight_layout()
    # 网格
    plt.grid(True)
    if output_dir is not None:
        plt.savefig(output_dir, dpi=400)
    # 显示图表
    plt.show()


def plot_C_VaR_and_returns_1(returns, VaR_series, CVaR_series, title='Returns and VaR (GARCH + GED)', xlabel='Time', ylabel='VaR', output_dir=None):
    """
    绘制收益率和VaR曲线在同一y轴上
    :param returns: 收益率序列
    :param title: 图片标题
    :param VaR_series: VaR序列
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    # 绘制收益率
    ax.plot(returns.index, returns, color='blue', label='Returns', alpha=0.6)
    # 绘制VaR曲线
    ax.plot(VaR_series.index, VaR_series, color='red', label='VaRs', alpha=0.6)
    ax.plot(CVaR_series.index, CVaR_series, color='red', label='CVaRs', alpha=0.6)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # 设置图例
    ax.legend(loc='best')
    # 标题
    plt.title(title)
    # 调整
    fig.tight_layout()
    # 网格
    plt.grid(True)
    if output_dir is not None:
        plt.savefig(output_dir, dpi=400)
    # 显示图表
    plt.show()


def plot_Vars_9599_returns(Returns, VaR_95, VaR_99, title, xlabel='Time', ylabel='Returns/VaR', output_dir=None):

    plt.figure(figsize=(20, 7))
    plt.plot(Returns,color='pink',label="Return Series")
    plt.plot(VaR_95, color='red', linestyle='--', label='95% VaR')
    plt.plot(VaR_99, color='blue', linestyle='--', label='99% VaR')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc='best')
    plt.grid(True)
    if output_dir:
        plt.savefig(output_dir, dpi=400)
    plt.show()


def calculate_and_visualize_correlation(Returns, VaR):

    # 计算收益率序列的绝对值
    absolute_returns = np.abs(Returns)
    # 计算相关系数
    correlation, _ = np.corrcoef(absolute_returns, VaR)
    # 绘制散点图
    plt.figure(figsize=(10, 6))
    plt.scatter(absolute_returns, VaR, alpha=0.5)
    plt.title('')
    plt.xlabel('Absolute Returns')
    plt.ylabel('VaR')
    plt.grid(True)
    plt.show()

    return correlation


def plot_boxplot(data, title, xlabel, ylabel, output_dir=None):
    """
    绘制箱线图，其中data为DataFrame，每一列绘制一个箱子。

    参数：
    - data: pd.DataFrame, 每一列绘制一个箱子
    - title: str, 图表标题
    - xlabel: str, x轴标签
    - ylabel: str, y轴标签
    - output_dir: str or None, 保存图表的目录路径。如果为 None，图表会直接显示。
    """
    # 检查 data 是否为 DataFrame
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data 参数必须是一个 pandas DataFrame 对象")

    # 创建图形和轴
    plt.figure(figsize=(10, 6))
    # 绘制箱线图
    data.boxplot()
    # 设置图表标题和标签
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc='best')
    # 调整布局
    plt.tight_layout()

    # 如果提供了输出目录，则保存图片
    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_path = os.path.join(output_dir, f"{title}.png")
        plt.savefig(output_path)
        print(f"图表已保存到: {output_path}")
    # 否则直接显示图片
    plt.show()
