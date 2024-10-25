# calculate_functions

## 定义了收益率的计算及参数法、等权重滑动窗口法、指数加权移动平均法、历史模拟法和GARCH模型计算Var和CVar

### **calculate_volatility_mean**

计算给定收益序列的标准差（波动率）和平均值

```
returns：收益率序列
```

### **calculate_returns**

计算每日对数收益率

```
prices：资产的价格序列
```

### **calculate_VaR_parameter**

使用参数方法来计算VaR

```
Returns：收益率序列
volatility：条件波动率
distribution：收益的分布类型，默认为"normal"（正态分布），其他可选项有"t"（t分布）和"ged"（广义误差分布）
lambda_ged：当使用GED分布时，需要指定形状参数
```

### **calculate_CVaR_parameter**

使用参数方法来计算CVaR

```
Returns：收益率序列
volatility：条件波动率
distribution：收益的分布类型
lambda_ged：GED分布的形状参数
```

### **calculate_VaR_CVaR_equ**

基于等权重滑动窗口来估计波动率，并进一步计算VaR或CVaR

```
Returns：收益率序列
winlen：滑动窗口长度，默认为50
distribution：收益的分布类型
lambda_ged：GED分布的形状参数
VaR_type：指定计算VaR还是CVaR，默认为'var'
```

### **calculate_VaR_CVaR_exp**

使用指数加权移动平均法（EWMA）来估计波动率，并计算VaR或CVaR

```
Returns：收益率序列
lambda_ewma：EWMA模型中的平滑因子，默认为0.94
distribution：收益的分布类型
lambda_ged：GED分布的形状参数
VaR_type：指定计算VaR还是CVaR
```
### **calculate_VaR_CVaR_garch**

利用GARCH模型来估计波动率，并计算VaR或CVaR

```
Returns：收益率序列
garch_type：GARCH模型的类型，默认为'garch'
distribution：收益的分布类型
lambda_ged：GED分布的形状参数
p：GARCH模型中自回归部分的阶数，默认为1
q：GARCH模型中移动平均部分的阶数，默认为1
VaR_type：指定计算VaR还是CVaR
```

### **calculate_VaR_CVaR_history**

通过历史模拟法来估计波动率，并计算VaR或CVaR

```
Returns：收益率序列
winlen：滑动窗口长度，默认为50
VaR_type：指定计算VaR还是CVaR
```
……  
……  
……  
……

# calculate_volatility

## 定义了收益率序列波动率计算的三种方法：GARCH、EWMA和等权重平均法

### es**timate_volatility_GARCH**

使用GARCH类模型估计收益率序列的条件波动率

```
returns：收益率序列
model_type：模型类型，可选择"GARCH"、"EGARCH"或"CGARCH"
> 如果model_type为"GARCH"，则使用普通的GARCH(p, q)模型
> 如果model_type为"EGARCH"，则使用扩展的GARCH模型（EGARCH），它可以捕捉波动率的不对称性
> 如果model_type为"CGARCH"，则使用GARCH(p, q)模型加上一个额外的参数来表示长期波动率成分
p：GARCH模型中的AR项数
q：GARCH模型中的MA项数
```

### **estimate_volatility_ewma**  
计算收益率序列的指数加权移动平均（EWMA）波动率
EWMA模型通过赋予过去的数据较少的权重来更新波动率估计
初始波动率可以通过收益率序列的前几期的标准差来估算

```
Returns：收益率序列
lambda_ewma：EWMA的衰减系数，默认为0.94
```

### **estimate_volatility_equ**  
计算收益率序列的等权重平均法波动率

```
Returns：收益率序列
winlen：计算波动率的窗口大小

```
……  
……  
……  
……
# Performance_evaluation

## 定义了九个评价指标
### **calculate_MAE**
计算每种VaR方法的平均绝对误差（MAE）

```
VaRs：DataFrame，每列代表一种VaR方法，每行代表某一天的数据
```
### **calculate_RMSE**
计算每种VaR方法的均方根相对偏差（RMSE）

```
VaRs：DataFrame，每列代表一种VaR方法，每行代表某一天的数据
```
### **calculate_APV**
计算每种VaR方法的日志收益率的年化波动百分比（APV）

```
VaRs：DataFrame，每列代表一种VaR方法，每行代表某一天的数据
```
### **calculate_and_visualize_coverage**
计算并可视化VaR的覆盖比率

```
Returns：收益率序列
VaR：VaR值序列
if_plot：是否绘制图表，True则绘制收益率序列与VaR覆盖区域的图表
```

### **calculate_and_visualize_correlation**
计算并可视化VaR与绝对收益率之间的相关性

```
Returns：收益率序列
VaR：VaR值序列
if_plot：是否绘制图表，True则绘制VaR与绝对收益率之间的散点图
```
### **tail_event_multiplier**
计算尾部事件中VaR与实际损失的平均比值

```
Returns：收益率序列
VaR：VaR值序列
if_plot：是否绘制图表，True则绘制尾部事件的比值图

```
### **max_tail_event_multiplier**
计算尾部事件中VaR与实际损失的最大比值


```
Rts：收益率序列
VaR：VaR值序列
if_plot：是否绘制图表，True则绘制尾部事件的最大比值图
```
### **calculate_scaled_mean_relative_bias**
计算调整到目标覆盖率后的平均相对偏差


```
Returns：收益率序列
VaRs：DataFrame，每列代表一种VaR方法的每日VaR值
target_coverage：目标覆盖率，默认为95%
```
……  
……  
……  
……
# VaR_models

## 定义了VaR计算绘图及评估的两个类
> 使用方法  
> 创建一个VaR实例，指定计算方法和其他参数  
> 使用calculate方法计算VaR和CVaR  
> 创建一个VaR_Evaluate实例，传入收益率序列  
> 使用evaluate方法评估VaR计算方法的表现
## **VaR 类**
封装了VaR计算的过程，可以根据不同的方法来计算VaR和CVaR，并提供绘图功能
### plot：绘制VaR图
### calculate：根据指定的方法计算VaR和CVaR，并根据if_plot参数决定是否绘制图表
```
ethod：指定使用的VaR计算方法，可以是"exp"（指数加权移动平均）、"equ"（等权重滑动窗口）、"para"/"parameter"（参数法，这里使用的是GARCH模型）或"his"/"history"（历史模拟法）
garch_type：指定GARCH模型的类型，默认为"GARCH"
winlen：滑动窗口长度，默认为250
lambda_ewma：EWMA模型中的平滑因子，默认为0.95
distribution：收益的分布类型，默认为"normal"
p：GARCH模型中自回归部分的阶数，默认为1
q：GARCH模型中移动平均部分的阶数，默认为1
VaR_type：指定计算VaR还是CVaR，默认为'var'
if_plot：是否绘制VaR图，默认为True
```

## **VaR_Evaluate 类**
用于评估多种VaR计算方法的表现

```

Returns：收益率序列
if_plot：是否绘制评估图表，默认为True
方法
evaluate：评估不同VaR计算方法的表现，并返回一个包含各种评估指标的DataFrame
VaRs：一个字典，键为VaR计算方法的名字，值为相应的VaR序列
confidence_level：置信水平，默认为0.95
评估指标
MAE：平均绝对误差
RMSE：均方根误差
APV：年化波动百分比
Coverage Ratio：覆盖率
Coverage_multiple：覆盖率调整倍数
Correlation：相关性
Average tail_event：尾部事件的平均乘数
Max tail_event：尾部事件的最大乘数
Relative_bias：调整后的平均相对偏差
```
……  
……  
……  
……
# VaR_ploting
定义了各种绘图函数
### **calculate_and_visualize_coverage**
计算并可视化收益率序列与VaR覆盖范围

```
Returns：收益率序列
VaR：VaR序列
```
### **plot_VaR_and_returns_0**
绘制收益率序列与VaR曲线在同一图表上

```
returns：收益率序列
VaR_series：VaR序列
title：图表标题
xlabel：x轴标签
ylabel：y轴标签
output_dir：输出目录路径
```
### **plot_C_VaR_and_returns_1**
绘制收益率序列与VaR及CVaR曲线在同一图表上

```
returns：收益率序列
VaR_series：VaR序列
CVaR_series：CVaR序列
title：图表标题
xlabel：x轴标签
ylabel：y轴标签
output_dir：输出目录路径
```
### **plot_Vars_9599_returns**
绘制收益率序列与95%和99%置信水平下的VaR曲线在同一图表上

```
Returns：收益率序列
VaR_95：95%置信水平下的VaR序列
VaR_99：99%置信水平下的VaR序列
title：图表标题
xlabel：x轴标签
ylabel：y轴标签
output_dir：输出目录路径
```
### **calculate_and_visualize_correlation**
计算并可视化收益率绝对值与VaR的相关性

```
Returns：收益率序列
VaR：VaR序列
```
### **plot_boxplot**
绘制给定数据的箱线图

```
data：数据框，每一列为一个箱子
title：图表标题
xlabel：x轴标签
ylabel：y轴标签
output_dir：输出目录路径
```

