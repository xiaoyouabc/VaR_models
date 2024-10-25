from VaR.VaR_models import VaR, VaR_Evaluate
from temp.Data_loader import Dataloader
import warnings
warnings.filterwarnings('ignore')


def get_twelve_VaRs(Returns, if_plot=False, VaR_type='var', output_prefix='VaR_'):
    """
        参考 <<Evaluation of Value-at-Risk Models Using Historical Data>>
        author: Darryll Hendricks
    :param Returns, pd.Series: 收益率数据
    :param if_plot, bool: 是否绘图
    :param VaR_type, str: 计算的VaR类型
    :return result_95, pd.DataFrame: 计算的12种95%置信水平的VaR值
    :return result_99, pd.DataFrame: 计算的12种99%置信水平的VaR值
    """
    VaRs = {}
    VaRs['equ_50'] = VaR(method='equ', winlen=50, if_plot=if_plot, VaR_type=VaR_type, output_dir=output_prefix + 'equ_50.png')
    VaRs['equ_125'] = VaR(method='equ', winlen=125, if_plot=if_plot, VaR_type=VaR_type, output_dir=output_prefix + 'equ_125.png')
    VaRs['equ_250'] = VaR(method='equ', winlen=250, if_plot=if_plot, VaR_type=VaR_type, output_dir=output_prefix + 'equ_250.png')
    VaRs['equ_500'] = VaR(method='equ', winlen=500, if_plot=if_plot, VaR_type=VaR_type, output_dir=output_prefix + 'equ_500.png')
    VaRs['equ_1250'] = VaR(method='equ', winlen=1250, if_plot=if_plot, VaR_type=VaR_type, output_dir=output_prefix + 'equ_1250.png')
    VaRs['his_125'] = VaR(method='his', winlen=125, if_plot=if_plot, VaR_type=VaR_type, output_dir=output_prefix + 'his_125.png')
    VaRs['his_250'] = VaR(method='his', winlen=250, if_plot=if_plot, VaR_type=VaR_type, output_dir=output_prefix + 'his_250.png')
    VaRs['his_500'] = VaR(method='his', winlen=500, if_plot=if_plot, VaR_type=VaR_type, output_dir=output_prefix + 'his_500.png')
    VaRs['his_1250'] = VaR(method='his', winlen=1250, if_plot=if_plot, VaR_type=VaR_type, output_dir=output_prefix + 'his_1250.png')
    VaRs['exp_0.94'] = VaR(method='exp', lambda_ewma=0.94, if_plot=if_plot, VaR_type=VaR_type, output_dir=output_prefix + 'exp_0.94.png')
    VaRs['exp_0.97'] = VaR(method='exp', lambda_ewma=0.97, if_plot=if_plot, VaR_type=VaR_type, output_dir=output_prefix + 'exp_0.97.png')
    VaRs['exp_0.99'] = VaR(method='exp', lambda_ewma=0.99, if_plot=if_plot, VaR_type=VaR_type, output_dir=output_prefix + 'exp_0.99.png')

    results_95 = {}
    results_99 = {}
    if VaR_type.lower() == 'var':
        var_type = 'VaR'
    elif VaR_type.lower() == 'cvar':
        var_type = 'CVaR'
    else:
        var_type = 'VaR'
    for name, model in VaRs.items():
        volatility_estimates, VaR95, VaR99 = model.calculate(Returns)
        results_95[f'{name}_{var_type}95'] = VaR95
        results_99[f'{name}_{var_type}99'] = VaR99

    return results_95, results_99


if __name__ == '__main__':

    Returns = Dataloader()
    results_95, results_99 = get_twelve_VaRs(Returns, VaR_type='var', output_prefix='output/pictures/VaR_')
    resultsc_95, resultsc_99 = get_twelve_VaRs(Returns, VaR_type='cvar', output_prefix='output/pictures/CVaR_')
    VaR_evaluate = VaR_Evaluate(Returns)
    df1 = VaR_evaluate.evaluate(results_95, confidence_level=0.95)
    df2 = VaR_evaluate.evaluate(results_99, confidence_level=0.99)
    # df1.to_excel("95.xlsx", index=True, engine='openpyxl')
    # df2.to_excel("99.xlsx", index=True, engine='openpyxl')
    #
    # cvarresults_95, cvarresults_99 = get_twelve_VaRs(Returns, VaR_type='cvar')
    # df1 = VaR_evaluate.evaluate(cvarresults_95, confidence_level=0.95)
    # df2 = VaR_evaluate.evaluate(cvarresults_99, confidence_level=0.99)
    # df1.to_excel("cvar95.xlsx", index=True, engine='openpyxl')
    # df2.to_excel("cvar99.xlsx", index=True, engine='openpyxl')
    #
    # tvarresults_95, tvarresults_99 = get_twelve_VaRs(Returns, VaR_type='var', distribution="t")
    # df1 = VaR_evaluate.evaluate(tvarresults_95, confidence_level=0.95)
    # df2 = VaR_evaluate.evaluate(tvarresults_99, confidence_level=0.99)
    # df1.to_excel("tvar95.xlsx", index=True, engine='openpyxl')
    # df2.to_excel("tvar99.xlsx", index=True, engine='openpyxl')
    #
    # tcvarresults_95, tcvarresults_99 = get_twelve_VaRs(Returns, VaR_type='cvar', distribution="t")
    # df1 = VaR_evaluate.evaluate(tcvarresults_95, confidence_level=0.95)
    # df2 = VaR_evaluate.evaluate(tcvarresults_99, confidence_level=0.99)
    # df1.to_excel("tcvar95.xlsx", index=True, engine='openpyxl')
    # df2.to_excel("tcvar99.xlsx", index=True, engine='openpyxl')
    #
    # gedvarresults_95, gedvarresults_99 = get_twelve_VaRs(Returns, VaR_type='var', distribution="ged")
    # df1 = VaR_evaluate.evaluate(gedvarresults_95, confidence_level=0.95)
    # df2 = VaR_evaluate.evaluate(gedvarresults_99, confidence_level=0.99)
    # df1.to_excel("gedvar95.xlsx", index=True, engine='openpyxl')
    # df2.to_excel("gedvar99.xlsx", index=True, engine='openpyxl')
    #
    # gedcvarresults_95, gedcvarresults_99 = get_twelve_VaRs(Returns, VaR_type='cvar', distribution="ged")
    # df1 = VaR_evaluate.evaluate(gedcvarresults_95, confidence_level=0.95)
    # df2 = VaR_evaluate.evaluate(gedcvarresults_99, confidence_level=0.99)
    # df1.to_excel("gedcvar95.xlsx", index=True, engine='openpyxl')
    # df2.to_excel("gedcvar99.xlsx", index=True, engine='openpyxl')

    garch_types = ['garch', 'egarch', 'cgarch']
    var_types = ['var', 'cvar']
    distributions = ['normal', 't', 'ged']

    results_95 = {}
    results_99 = {}

    for var_type in var_types:
        for garch_type in garch_types:
            for distribution in distributions:
                Var_model = VaR(method='para', if_plot=True, VaR_type=var_type, distribution=distribution)
                _, VaR_95, VaR_99 = Var_model.calculate(Returns.dropna())
                if var_type == 'cvar' and distribution == 'ged':
                    continue
                results_95[f'{garch_type}_{distribution}_{var_type}95'] = VaR_95
                results_99[f'{garch_type}_{distribution}_{var_type}99'] = VaR_99
    keys = list(results_95.keys())
    df1 = VaR_evaluate.evaluate(results_95, confidence_level=0.95)
    df2 = VaR_evaluate.evaluate(results_99, confidence_level=0.99)
    df1.to_excel("output/garch_distribution_vartype95.xlsx", index=True, engine='openpyxl')
    df2.to_excel("output/garch_distribution_vartype99.xlsx", index=True, engine='openpyxl')
