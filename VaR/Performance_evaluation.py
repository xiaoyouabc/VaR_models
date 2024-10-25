# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# 假设Rts是收益率的数组

def calculate_MAE(VaRs):
    VaRs = VaRs.dropna()
    # 第一步：计算每日12种方法的平均VaR值
    average_VaR = VaRs.mean(axis=1)
    # 第二步：计算每日的相对偏差（每种方法与平均VaR的百分比差异）
    for col in VaRs.columns:
        VaRs[col] = (VaRs[col] - average_VaR)/average_VaR

    # 第三步：计算每种方法的平均相对偏差（在所有日期上的平均值）
    mean_relative_bias = VaRs.mean(axis=0)
    return mean_relative_bias


def calculate_RMSE(VaRs):
    """
    计算每种VaR方法的均方根相对偏差（RMSE）。
    参数:
    VaRs - DataFrame，每列代表一种VaR方法，每行代表某一天的数据。
    返回:
    rmse - 每种VaR方法的均方根相对偏差（RMSE），以Series形式返回。
    """
    VaRs = VaRs.dropna(axis=0, how='any')
    # 第一步：计算每日的平均VaR值
    avg_VaR = VaRs.mean(axis=1)
    for col in VaRs.columns:
        VaRs[col] = (VaRs[col] - avg_VaR)/avg_VaR
    # 第三步：计算每日相对偏差的平方
    squared_bias = VaRs ** 2
    # 第四步：对所有日期的平方相对偏差求平均
    mean_squared_bias = squared_bias.mean(axis=0)
    # 第五步：开平方，得到RMSE
    rmse = np.sqrt(mean_squared_bias)
    return rmse


def calculate_APV(VaRs):
    """
    计算每种VaR方法的年化波动百分比（APV）。
    参数:
    VaRs - DataFrame，每列代表一种VaR方法，每行代表某一天的数据。
    返回:
    apv - 每种VaR方法的年化波动百分比（APV），以Series形式返回。
    """
    # 第一步：计算每日VaR的百分比变化
    daily_changes = np.log(VaRs / VaRs.shift(1)).dropna()
    # 第二步：计算每日百分比变化的标准差
    daily_std = daily_changes.std()
    # 第三步：将标准差年化
    annualized_volatility = daily_std * np.sqrt(250)
    return annualized_volatility


def calculate_and_visualize_coverage(Returns, VaR, if_plot):
    # 计算投资组合价值的损失
    losses = Returns
    # 确定损失是否小于或等于VaR值
    min_length = min(len(losses), len(VaR))
    totallen = len(losses)
    losses = losses[totallen - min_length:]
    print(len(losses))
    coverage = ( -VaR<=losses )
    # 计算所涵盖结果的比例
    coverage_ratio = np.mean(coverage)

    if if_plot:
        plt.figure(figsize=(30, 15))

        plt.plot(range(len(losses)), Returns[totallen - min_length:], label='收益率序列')
        plt.axhline(y=np.mean(VaR), color='r', linestyle='--', label='平均VaR')
        plt.fill_between(range(len(losses)), VaR, color='g', alpha=0.3, label='VaR覆盖区域')
        plt.scatter(range(len(losses)), losses, color='pink', label='损失')
        plt.title('收益率序列与VaR覆盖')
        plt.xlabel('时间点')
        plt.ylabel('收益率/损失')
        plt.legend()
        plt.show()

    return coverage_ratio



def calculate_and_visualize_correlation(Returns, VaR, if_plot):
    # 计算收益率序列的绝对值
    absolute_returns = np.abs(Returns).dropna()
    # min_length = min(len(absolute_returns), len(VaR))
    if len(absolute_returns) >= VaR.shape[0]:
        absolute_returns = absolute_returns[absolute_returns.index.isin(VaR.index)]
        VaR1 = VaR.copy()
    else:
        VaR1 = VaR[VaR.index.isin(absolute_returns.index)]
    # absolute_returns = absolute_returns[len(VaR) - min_length:]
    # VaR = VaR[len(VaR) - min_length:]
    # 计算相关系数
    correlation, _ = np.corrcoef(absolute_returns, VaR1)

    # 绘制散点图
    if if_plot:
        plt.figure(figsize=(10, 6))
        plt.scatter(absolute_returns, VaR1, alpha=0.5)
        plt.title('')
        plt.xlabel('Absolute Returns')
        plt.ylabel('VaR')
        plt.grid(True)
        plt.show()

    return correlation


def tail_event_multiplier(Returns, VaR, if_plot):
    # 计算每个时间点的损失
    losses = np.abs(Returns)
    min_length = min(len(losses), len(VaR))

    losses = losses[len(losses) - min_length:]

    # 确定尾部事件（即损失大于VaR的情况）
    tail_events = losses > VaR

    # 如果VaR为0，则设置比值为np.nan，否则计算比值
    tail_event_multipliers = np.where(VaR != 0, losses / VaR, np.nan)

    # 移除尾部事件的平均倍数，忽略np.nan值
    average_multiplier = np.nanmean(tail_event_multipliers[tail_events])

    if if_plot:
        plt.figure(figsize=(10, 6))
        # 只绘制尾部事件的点
        plt.scatter(range(len(tail_event_multipliers)), tail_event_multipliers, color='red',
                    label='Tail Event Multipliers', alpha=0.6, s=8)
        # 绘制平均倍数线，如果average_multiplier是np.nan，则不绘制
        if not np.isnan(average_multiplier):
            plt.axhline(y=average_multiplier, color='blue', linestyle='--',
                        label=f'Average Multiplier: {average_multiplier:.2f}')
        plt.title('Tail Event Multipliers vs VaR')
        plt.xlabel('Time')
        plt.ylabel('Multiplier')
        plt.legend()
        plt.grid(True)
        plt.show()

    return average_multiplier


def max_tail_event_multiplier(Rts, VaR, if_plot):
    # 计算每个时间点的损失
    losses = np.abs(Rts)
    min_length = min(len(losses), len(VaR))

    losses = losses[len(losses) - min_length:]

    # 确定尾部事件（即损失大于VaR的情况）
    tail_events = losses > VaR

    # 如果VaR为0，则设置比值为np.nan，否则计算比值
    tail_event_multipliers = np.where(VaR != 0, losses / VaR, np.nan)

    # 移除尾部事件的平均倍数，忽略np.nan值
    max_multiplier = np.nanmax(tail_event_multipliers[tail_events])

    if if_plot:
        plt.figure(figsize=(10, 6))
        # 只绘制尾部事件的点
        plt.scatter(range(len(tail_event_multipliers)), tail_event_multipliers, color='red',
                    label='Tail Event Multipliers', alpha=0.6, s=8)
        # 绘制平均倍数线，如果average_multiplier是np.nan，则不绘制
        if not np.isnan(max_multiplier):
            plt.axhline(y=max_multiplier, color='blue', linestyle='--', label=f'Max Multiplier: {max_multiplier:.2f}')
        plt.title('Tail Event Multipliers vs VaR')
        plt.xlabel('Time')
        plt.ylabel('Multiplier')
        plt.legend()
        plt.grid(True)
        plt.show()

    return max_multiplier


def calculate_scaled_mean_relative_bias(Returns, VaRs, target_coverage=0.95):
    """
    计算调整到目标覆盖率后的平均相对偏差。
    参数:
    losses - DataFrame，每行表示一个日期的实际损失。
    VaRs - DataFrame，每列表示一种VaR方法的每日VaR值。
    target_coverage - 目标覆盖率（默认为95%）。
    返回:
    mean_relative_bias - 每种方法的平均相对偏差，Series格式。
    """
    ori_VaRs = VaRs.copy()
    for col in VaRs.columns:
        VaRs[col] = np.where(VaRs[col].dropna() <= Returns[Returns.index.isin(VaRs[col].dropna().index)], 1, 0)
    # 第一步：计算每种方法的实际覆盖率（实际损失 <= VaR 的比例）
    coverage = VaRs.mean(axis=0)
    # 第二步：计算调整倍数
    multiples = target_coverage / coverage
    # 第三步：调整VaR值
    scaled_VaRs = ori_VaRs.mul(multiples, axis=1)
    # 第四步：计算每日的平均VaR值（所有方法的平均）
    avg_scaled_VaR = scaled_VaRs.mean(axis=1)
    for col in scaled_VaRs.columns:
        scaled_VaRs[col] = (scaled_VaRs[col] - avg_scaled_VaR) / avg_scaled_VaR
    # 第六步：计算所有日期的平均相对偏差
    mean_relative_bias = scaled_VaRs.mean(axis=0)
    return mean_relative_bias