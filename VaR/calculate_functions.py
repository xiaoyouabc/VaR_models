# -*- coding: utf-8 -*-

import numpy as np
from scipy import stats as stats
import pandas as pd
from VaR.calculate_volatility import estimate_volatility_GARCH, estimate_volatility_equ, estimate_volatility_ewma
import warnings
warnings.filterwarnings('ignore')


def calculate_volatility_mean(returns):
    return returns.std(), returns.mean()


def calculate_returns(prices):
    """
    计算每日收益率
    """
    returns = np.log(prices / prices.shift(1))
    return returns.dropna()


def calculate_VaR_parameter(Returns, volatility, distribution="normal", lambda_ged=1.5):
    """
    参数法计算 VaR
    :param Returns: 收益率序列
    :param volatility: 条件波动率
    :param confidence_level: 置信水平
    :param distribution: 分布类型 ("normal", "t", "ged")
    :param lambda_ged: GED 分布的形状参数
    :return: VaR 值序列
    """
    confidence_level1 = 0.95
    confidence_level2 = 0.99
    mean_return = np.mean(Returns)  # 计算收益率的均值
    if distribution == "normal":
        # 使用正态分布计算分位点
        z = stats.norm.ppf(1 - confidence_level1)  # 正态分布的分位点
        VaR_95 = mean_return + z * volatility  # VaR = 均值 + 分位点 * 波动率
        z = stats.norm.ppf(1 - confidence_level2)  # 正态分布的分位点
        VaR_99 = mean_return + z * volatility  # VaR = 均值 + 分位点 * 波动率
    elif distribution == "t":
        # 使用 t 分布计算分位点
        df = len(Returns) - 1  # t 分布的自由度
        t_critical = stats.t.ppf(1 - confidence_level1, df=df)
        VaR_95 = mean_return + t_critical * volatility  # VaR = 均值 + t 分布分位点 * 波动率
        t_critical = stats.t.ppf(1 - confidence_level2, df=df)
        VaR_99 = mean_return + t_critical * volatility  # VaR = 均值 + t 分布分位点 * 波动率
    elif distribution == "ged":
        # 使用广义误差分布（GED）计算分位点
        ged_critical = stats.gennorm.ppf(1 - confidence_level1, beta=lambda_ged)
        VaR_95 = mean_return + ged_critical * volatility  # VaR = 均值 + GED 分布分位点 * 波动率
        ged_critical = stats.gennorm.ppf(1 - confidence_level2, beta=lambda_ged)
        VaR_99 = mean_return + ged_critical * volatility  # VaR = 均值 + GED 分布分位点 * 波动率
    else:
        raise ValueError("Unsupported distribution type. Choose 'normal', 't', or 'ged'.")
    # VaR 通常为负值，表示损失，返回绝对值
    VaR_95 = pd.Series(VaR_95, index=Returns.index)
    VaR_99 = pd.Series(VaR_99, index=Returns.index)
    return VaR_95.dropna(), VaR_99.dropna()


def calculate_CVaR_parameter(Returns, volatility, distribution="normal", lambda_ged=1.5):
    """

    :param Returns:
    :param volatility:
    :param distribution:
    :param lambda_ged:
    :return:
    """
    confidence_level1 = 0.95
    confidence_level2 = 0.99
    mean_return = np.mean(Returns)  # 计算收益率的均值
    # 根据均值和波动率拟合不同的分布
    if distribution == "normal":
        VaR_95 = stats.norm.ppf(1 - confidence_level1)
        CVaR_95 = mean_return - volatility * (stats.norm.pdf(VaR_95) / (1 - confidence_level1))
        VaR_99 = stats.norm.ppf(1 - confidence_level2)
        CVaR_99 = mean_return - volatility * (stats.norm.pdf(VaR_99) / (1 - confidence_level2))

    elif distribution == "t":
        # 拟合 t 分布，并计算 VaR
        df = len(Returns) - 1
        # t 分布的 CVaR 计算
        VaR_95_t = stats.t.ppf(1 - confidence_level1, df)
        VaR_99_t = stats.t.ppf(1 - confidence_level2, df)
        CVaR_95 = mean_return - volatility * (
                (df + VaR_95_t ** 2) / (df - 1) * stats.t.pdf(VaR_95_t, df) / (1 - confidence_level1)
        )
        CVaR_99 = mean_return - volatility * (
                (df + VaR_99_t ** 2) / (df - 1) * stats.t.pdf(VaR_99_t, df) / (1 - confidence_level2)
        )

    elif distribution == "ged":
        # 拟合 GED 分布，并计算 VaR
        fitted_dist = stats.gennorm(beta=lambda_ged, loc=mean_return, scale=volatility)
        VaR_95_ged = fitted_dist.ppf(1 - confidence_level1)
        VaR_99_ged = fitted_dist.ppf(1 - confidence_level2)
        CVaR_95 = mean_return - volatility * (
                fitted_dist.pdf(VaR_95_ged) / (1 - confidence_level1)
        )
        CVaR_99 = mean_return - volatility * (
                fitted_dist.pdf(VaR_99_ged) / (1 - confidence_level2)
        )
    else:
        raise ValueError("Unsupported distribution type. Choose 'normal', 't', or 'ged'.")
    CVaR_95 = pd.Series(CVaR_95, index=Returns.index)
    CVaR_99 = pd.Series(CVaR_99, index=Returns.index)
    return CVaR_95.dropna(), CVaR_99.dropna()


def calculate_VaR_CVaR_equ(Returns, winlen=50, distribution="normal", lambda_ged=1.5, VaR_type='var'):
    # 初始化波动率估计数组
    volatility_estimates = estimate_volatility_equ(Returns, winlen)
    if VaR_type.lower() == 'var':
        VaR_95, VaR_99 = calculate_VaR_parameter(Returns, volatility_estimates, distribution=distribution, lambda_ged=lambda_ged)
    elif VaR_type.lower() == 'cvar':
        VaR_95, VaR_99 = calculate_CVaR_parameter(Returns, volatility_estimates, distribution=distribution, lambda_ged=lambda_ged)
    else:
        print('目前仅支持计算VaR和CVaR')
        print('请重新确认Var_type参数')
        return None, None, None
    VaR_95 = VaR_95[winlen:]
    VaR_99 = VaR_99[winlen:]
    volatility_estimates = volatility_estimates[winlen:]
    return volatility_estimates.dropna(), VaR_95, VaR_99


def calculate_VaR_CVaR_exp(Returns, lambda_ewma=0.94, distribution="normal", lambda_ged=1.5, VaR_type='var'):
    """

    :param Returns:
    :param lambda_ewma:
    :param distribution:
    :param lambda_ged:
    :return:
    """
    volatility_estimates = estimate_volatility_ewma(Returns, lambda_ewma)
    if VaR_type.lower() == 'var':
        VaR_95, VaR_99 = calculate_VaR_parameter(Returns, volatility_estimates, distribution=distribution, lambda_ged=lambda_ged)
    elif VaR_type.lower() == 'cvar':
        VaR_95, VaR_99 = calculate_CVaR_parameter(Returns, volatility_estimates, distribution=distribution, lambda_ged=lambda_ged)
    else:
        print('目前仅支持计算VaR和CVaR')
        print('请重新确认Var_type参数')
        return None, None, None
    return volatility_estimates.dropna(), VaR_95, VaR_99


def calculate_VaR_CVaR_garch(Returns, garch_type='garch', distribution="normal", lambda_ged=1.5, p=1, q=1, VaR_type='var'):
    """

    :param Returns:
    :param garch_type:
    :param distribution:
    :param lambda_ged:
    :param var_type:
    :return:
    """
    volatility_estimates = estimate_volatility_GARCH(Returns, garch_type, p=p, q=q)
    if VaR_type.lower() == 'var':
        VaR_95, VaR_99 = calculate_VaR_parameter(Returns, volatility_estimates, distribution=distribution, lambda_ged=lambda_ged)
    elif VaR_type.lower() == 'cvar':
        VaR_95, VaR_99 = calculate_CVaR_parameter(Returns, volatility_estimates, distribution=distribution, lambda_ged=lambda_ged)
    else:
        print('目前仅支持计算VaR和CVaR')
        print('请重新确认Var_type参数')
        return None, None, None
    return volatility_estimates.dropna(), VaR_95, VaR_99


def calculate_VaR_CVaR_history(Returns, winlen=50, VaR_type='var'):
    """
    根据滚动窗口计算历史模拟法的 VaR 值
    :param prices: 资产价格的时间序列数据（数组或 Series）
    :param winlen: 回溯窗口期（天数）
    :param confidence_level: VaR 置信水平 (如 0.95, 0.99)
    :return: VaR 序列
    """
    confidence_level1 = 0.95
    confidence_level2 = 0.99
    volatility_estimates = pd.Series(index=Returns.index[winlen:], name='volatility')
    if VaR_type.lower() == 'var':
        var_series_95 = pd.Series(index=Returns.index[winlen:], name='VaR_95')
        var_series_99 = pd.Series(index=Returns.index[winlen:], name='VaR_99')

        for i in range(winlen, len(Returns)):
            window_returns = Returns[i - winlen:i]
            var_series_95.iloc[i - winlen] = np.percentile(window_returns, (1 - confidence_level1) * 100)
            var_series_99.iloc[i - winlen] = np.percentile(window_returns, (1 - confidence_level2) * 100)
            volatility_estimates.iloc[i - winlen] = np.std(window_returns)
        return volatility_estimates.dropna(), var_series_95.dropna(), var_series_99.dropna()
    elif VaR_type.lower() == 'cvar':
        cvar_95 = pd.Series(index=Returns.index[winlen:], name='CVaR_95')
        cvar_99 = pd.Series(index=Returns.index[winlen:], name='CVaR_99')
        for i in range(winlen, len(Returns)):
            window_returns = Returns[i - winlen:i]
            var_value_95 = np.percentile(window_returns, (1 - confidence_level1) * 100)
            cvar_95.iloc[i - winlen] = window_returns[window_returns <= var_value_95].mean()
            var_value_99 = np.percentile(window_returns, (1 - confidence_level2) * 100)
            cvar_99.iloc[i - winlen] = window_returns[window_returns <= var_value_99].mean()
            volatility_estimates.iloc[i - winlen] = np.std(window_returns)
        return volatility_estimates.dropna(), cvar_95.dropna(), cvar_99.dropna()

    else:
        raise ValueError("Invalid VaR_type. Choose 'var' or 'cvar'.")




