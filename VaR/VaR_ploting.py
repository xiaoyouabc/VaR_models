# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import os
from scipy.stats import gennorm
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



def plot_four_VaRs_returns(Returns, VaR_95, VaR_99, CVaR_95, CVaR_99,
                           title, xlabel='Time', ylabel='Returns/VaR/CVaR',
                           output_dir=None):
    """
    绘制返回序列与 95% 和 99% VaR / CVaR 的对比图。
    参数：
    Returns: 序列 - 返回序列
    VaR_95: 序列 - 95% VaR 序列
    VaR_99: 序列 - 99% VaR 序列
    CVaR_95: 序列 - 95% CVaR 序列
    CVaR_99: 序列 - 99% CVaR 序列
    title: str - 图表标题
    xlabel: str - X 轴标签
    ylabel: str - Y 轴标签
    output_dir: str - 如果提供，则将图表保存到指定路径
    """
    plt.figure(figsize=(20, 7))
    # 绘制返回序列
    plt.plot(Returns, color='pink', label='Return Series')
    # 绘制 VaR 线
    plt.plot(VaR_95, color='red', linestyle='--', label='95% VaR')
    plt.plot(VaR_99, color='blue', linestyle='--', label='99% VaR')
    # 绘制 CVaR 线
    plt.plot(CVaR_95, color='orange', linestyle='-', label='95% CVaR')
    plt.plot(CVaR_99, color='purple', linestyle='-', label='99% CVaR')
    # 设置标题和标签
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # 显示图例和网格
    plt.legend(loc='best')
    plt.grid(True)
    # 如果指定了输出路径，则保存图片
    if output_dir:
        plt.savefig(output_dir, dpi=400)

    # 显示图表
    plt.show()


def plot_multiple_ged_distributions(betas, sample_size=1000, xlim=(-5, 5), output_dir=None):
    """
    绘制多个 beta 值的 GED 分布在一张图上。

    参数：
    betas: list - 多个 beta 形状参数的列表。
    sample_size: int - 每个分布的样本数量，默认1000。
    xlim: tuple - X 轴的范围，默认为 (-5, 5)。
    output_dir: str - 如果提供路径，将图保存为文件。
    """
    plt.figure(figsize=(12, 8))

    # 遍历 beta 列表，逐个绘制 PDF 曲线
    for beta in betas:
        # 生成数据并计算 PDF
        data = gennorm.rvs(beta, size=sample_size)
        x = np.linspace(*xlim, 1000)  # 限制 X 轴范围
        pdf = gennorm.pdf(x, beta)

        # 绘制 PDF 曲线
        plt.plot(x, pdf, lw=2, label=f'beta = {beta}')

    # 设置标题和标签
    plt.title('GED Distributions with Different Beta Values', fontsize=16)
    plt.xlabel('Value', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend(loc='best')
    plt.grid(True)
    plt.xlim(xlim)  # 设置 X 轴范围

    # 保存图像（如果提供路径）
    if output_dir:
        plt.savefig(output_dir, dpi=300)

    # 显示图表
    plt.show()


def plot_3x3_subplots_from_dfs(df1, df2, output_dir=None, rotation=45):
    """
    绘制 3x3 子图，每个子图展示两个 DataFrame 中对应列的数据。

    参数：
    df1: DataFrame - 第一个 DataFrame，包含至少 9 列。
    df2: DataFrame - 第二个 DataFrame，包含至少 9 列。
    output_dir: str - 如果提供路径，将图保存为文件。
    rotation: int - X 轴和 Y 轴标签的旋转角度，默认 45 度。
    """
    if df1.shape[1] < 9 or df2.shape[1] < 9:
        raise ValueError("Both DataFrames must contain at least 9 columns.")

    fig, axes = plt.subplots(3, 3, figsize=(15, 15))  # 创建 3x3 子图网格
    fig.suptitle('Comparison of VaRs and CVaRs', fontsize=20)  # 整体标题

    # 遍历 3x3 子图，将每对列数据绘制在子图上
    for i in range(9):
        ax = axes[i // 3, i % 3]  # 获取当前子图的轴对象

        # 选择第 i 列的数据
        data1 = df1.iloc[:, i]
        data2 = df2.iloc[:, i]

        # 绘制两条曲线
        ax.plot(data1, label=f'{df1.columns[i]} (VaR)', color='blue', linestyle='--')
        ax.plot(data2, label=f'{df2.columns[i]} (CVaR)', color='green', linestyle='-')

        # 设置标题和美化子图
        ax.set_title(f'{df1.columns[i]} vs {df2.columns[i]}', fontsize=14)
        ax.set_xlabel('Index', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.legend(loc='best')
        ax.grid(alpha=0.3)

        # 设置 X 轴和 Y 轴标签的旋转角度
        for tick in ax.get_xticklabels():
            tick.set_rotation(rotation)
        for tick in ax.get_yticklabels():
            tick.set_rotation(rotation)

    # 调整子图之间的间距
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # 保存图像（如果提供路径）
    if output_dir:
        plt.savefig(output_dir, dpi=300)

    # 显示图表
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


