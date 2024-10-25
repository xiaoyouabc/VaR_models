# -*- coding: utf-8 -*-

from VaR.calculate_functions import (calculate_VaR_CVaR_exp, calculate_VaR_CVaR_equ, calculate_VaR_CVaR_garch,
                                 calculate_VaR_CVaR_history)
import pandas as pd
from VaR.Performance_evaluation import (calculate_MAE, calculate_RMSE, calculate_APV, calculate_and_visualize_correlation,
                                     calculate_and_visualize_coverage, tail_event_multiplier,
                                    max_tail_event_multiplier, calculate_scaled_mean_relative_bias)
from VaR.VaR_ploting import plot_Vars_9599_returns
import warnings
warnings.filterwarnings('ignore')


class VaR:
    def __init__(self, method, garch_type="GARCH", winlen=250, lambda_ewma=0.95, p=1, q=1, distribution="normal",
                 VaR_type='var', if_plot=True, output_dir=None):
        super().__init__()
        self.method = method
        self.PARA_model_type = garch_type
        self.window_size = winlen
        self.lambda_ewma = lambda_ewma
        self.distribution = distribution
        self.p = p
        self.q = q
        self.VaR_type = VaR_type
        self.if_plot = if_plot
        self.output_dir = output_dir

    def plot(self, Returns, VaR_95, VaR_99):

        plot_Vars_9599_returns(Returns, VaR_95, VaR_99, title=f'{self.method.title()} VaR 95%|99%置信水平',
                               output_dir=self.output_dir)

    def calculate(self, Returns):
        # volatility_estimates = None
        if self.method.lower() in ['exp']:
            volatility_estimates, VaR95, VaR99 = calculate_VaR_CVaR_exp(Returns, self.lambda_ewma,
                                                                        VaR_type=self.VaR_type)
        elif self.method.lower() in ['equ']:
            volatility_estimates, VaR95, VaR99 = calculate_VaR_CVaR_equ(Returns, self.window_size,
                                                                        VaR_type=self.VaR_type)
        elif self.method.lower() in ['para', 'parameter']:
            volatility_estimates, VaR95, VaR99 = calculate_VaR_CVaR_garch(Returns, garch_type="GARCH", p=self.p,
                                                                          q=self.q,
                                                                          distribution=self.distribution,
                                                                          VaR_type=self.VaR_type)
        elif self.method.lower() in ["his", 'history']:
            volatility_estimates, VaR95, VaR99 = calculate_VaR_CVaR_history(Returns, self.window_size,
                                                                            VaR_type=self.VaR_type)
        else:
            print('目前仅支持')
            return None, None, None

        if self.if_plot:
            self.plot(Returns, VaR95, VaR99)
        return volatility_estimates, VaR95, VaR99


class VaR_Evaluate():
    def __init__(self, Returns, if_plot=False):
        super().__init__()
        self.Returns = Returns

        self.if_plot = if_plot

    def evaluate(self, VaRs, confidence_level=0.95):

        all_VaRs = pd.concat(VaRs, axis=1).dropna()
        all_VaRs.columns = list(VaRs.keys())
        MAE = calculate_MAE(all_VaRs.copy())
        RMSE = calculate_RMSE(all_VaRs.copy())
        APV = calculate_APV(all_VaRs.copy())
        relative_bias = calculate_scaled_mean_relative_bias(self.Returns.copy(), all_VaRs.copy(), target_coverage=confidence_level)
        # 初始化用于存储每种方法的结果的列表
        list1, list2, list3, list4, list5 = [], [], [], [], []

        # 遍历每种VaR方法，计算其他性能指标
        for name, VaR in VaRs.items():
            coverage_ratio = calculate_and_visualize_coverage(self.Returns, -VaR, if_plot=self.if_plot)
            coverage_multiple = confidence_level / coverage_ratio
            correlation = calculate_and_visualize_correlation(self.Returns, -VaR, if_plot=self.if_plot)
            average_multiplier = tail_event_multiplier(self.Returns, -VaR, if_plot=self.if_plot)
            max_multiplier = max_tail_event_multiplier(self.Returns, -VaR, if_plot=self.if_plot)

            # 将结果添加到对应的列表中
            list1.append(coverage_ratio)
            list2.append(correlation[1])
            list3.append(average_multiplier)
            list4.append(max_multiplier)
            list5.append(coverage_multiple)
        # 将所有结果整理为DataFrame
        results = pd.DataFrame({
            'MAE': MAE,
            'RMSE': RMSE,
            'APV': APV,
            'Coverage Ratio': list1,
            'Coverage_multiple': list5,
            'Correlation': list2,
            'Average tail_event': list3,
            'Max tail_event': list4,
            'Relative_bias': relative_bias
        }, index=list(VaRs.keys()))

        return results










