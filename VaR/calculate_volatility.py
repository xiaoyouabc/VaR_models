# -*- coding: utf-8 -*-

from arch import arch_model
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


def estimate_volatility_GARCH(returns, model_type="GARCH", p=1, q=1):
    """
    使用 GARCH, EGARCH, CGARCH 模型估计收益的条件波动率
    :param returns: 收益率序列
    :param model_type: 模型类型 ("GARCH", "EGARCH", "CGARCH")
    :param p: GARCH 模型中的 AR 项数
    :param q: GARCH 模型中的 MA 项数
    :return: 条件波动率序列
    """
    if model_type.lower() == "garch":
        model = arch_model(returns, vol='GARCH', p=p, q=q)
    elif model_type.lower() == "egarch":
        model = arch_model(returns, vol='EGARCH', p=p, q=q)
    elif model_type.lower() == "cgarch":
        model = arch_model(returns, vol='GARCH', p=p, q=q, o=1)  # CGARCH 使用的模型是带有波动率长期成分
    else:
        raise ValueError("Unsupported model type. Choose 'GARCH', 'EGARCH', or 'CGARCH'.")
    # 拟合模型
    res = model.fit(disp="off")
    # 返回条件波动率
    return res.conditional_volatility


def estimate_volatility_ewma(Returns, lambda_ewma=0.94):
    """
    计算收益率序列的EWMA波动率。
    参数：
    Returns : pd.Series
        收益率序列。
    lambda_ewma : float, 可选
        EWMA的衰减系数，默认值为0.94。
    返回：
    volatility_estimates : pd.Series
        计算得到的EWMA波动率序列。
    """
    # 初始化波动率估计数组
    volatility_estimates = np.zeros(len(Returns))
    # 建议使用非零初始波动率
    volatility_estimates[0] = Returns[:5].std()  # 可以用前5期的标准差来初始化

    # 计算EWMA波动率
    for i in range(1, len(Returns)):
        volatility_estimates[i] = np.sqrt(lambda_ewma * volatility_estimates[i-1]**2 +
                                          (1 - lambda_ewma) * (Returns.iloc[i])**2)
    return pd.Series(volatility_estimates, index=Returns.index)


def estimate_volatility_equ(Returns, winlen):
    """
    计算收益率序列的等权重平均法波动率。
    参数：
    Returns : pd.Series
        收益率序列。
    window_size : int
        计算波动率的窗口大小。
    返回：
    volatility_estimates : pd.Series
        计算得到的等权重波动率序列。
    """
    # 初始化波动率估计数组
    volatility_estimates = np.zeros(len(Returns))

    # 计算等权重平均法波动率
    for i in range(winlen, len(Returns)):
        window = Returns[i - winlen:i]
        n = (window - window.mean()) ** 2
        volatility_estimates[i] = np.sqrt(n.sum() / (winlen - 1))

    # 由于前window_size-1个值没有足够的数据来计算标准差，我们可以将它们设置为第一个波动率估计值
    volatility_estimates[:winlen - 1] = volatility_estimates[winlen - 1]

    return pd.Series(volatility_estimates, index=Returns.index)

