# -*- coding: utf-8 -*-

import numpy as np
from VaR.calculate_functions import calculate_returns, calculate_VaR_CVaR_garch

import warnings
warnings.filterwarnings('ignore')


if __name__ == '__main__':
    # 假设有1000天的股票价格数据
    np.random.seed(42)
    prices = Dataloader()
    # 计算收益率
    returns = calculate_returns(prices)

    # 估计波动率 (使用 GARCH 模型为例)
    # garch_volatility = estimate_volatility_GARCH(returns, model_type="GARCH")

    # 使用估计的波动率计算VaR，选择 GED 分布
    _, VaR_95, VaR_99 = calculate_VaR_CVaR_garch(returns, distribution="ged", lambda_ged=1.5)
    _, CVaR_95, CVaR_99 = calculate_VaR_CVaR_garch(returns, distribution="ged", lambda_ged=1.5, VaR_type='cvar')
    # plot_VaR_and_returns_0(returns, VaR_series, title='Returns and VaR (GARCH + GED)')
    # plot_VaR_and_returns_0(returns, VaR_series, title='Returns and VaR (GARCH + GED)')
    # 绘制收益率和VaR曲线
    # plot_VaR_and_returns_0(returns, CVaR_series, title='Returns and CVaR (GARCH + GED)')

    # plot_C_VaR_and_returns_1(returns, VaR_series, CVaR_series, title='Returns and Var, CVaR (GARCH + GED)')

