# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


def Dataloader(path='test/VARData.xlsx', output_dir=None):
    """
        载入原始数据并简单处理
    :param path, str: 原始数据的路径
    :param output_dir, str: 存储路径
    :return prices, pd.DataFrame: 价格数据表格
    """
    data1 = pd.read_excel(path, index_col=0, sheet_name=0)
    data1=data1.drop('日期',axis=1)
    data1.dropna(inplace=True)
    data2 = pd.read_excel(path, index_col=0, sheet_name=1)
    data2 = data2.drop('日期', axis=1)
    data2.rename(columns={data2.columns[0]: 'Value'}, inplace=True)
    data2['Weight']= data2['Value'] / data2['Value'].sum()
    prices = pd.Series(index=data1.columns, name=' prices')
    for i in data1.columns:
        prices[i] = (data1[i] * data2['Weight']).sum(axis=0)
    Returns = np.log(prices / prices.shift(1))
    if output_dir:
        Returns.to_pickle(output_dir)
    return Returns



