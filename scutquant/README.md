## 快速上手  
请查看[tutorial.ipynb](https://github.com/HaoningChen/ScutQuant/blob/main/%E5%AE%9E%E8%B7%B5%E6%A1%88%E4%BE%8B/tutorial.ipynb)以快速上手  

## 各模块介绍(按照量化研究的流程介绍)  

### [data](https://github.com/HaoningChen/ScutQuant/blob/main/scutquant/data.py):   
获取数据模块, 基于tushare实现. 目前功能只有获取指数成分股数据   
### [operators](https://github.com/HaoningChen/scutquant/blob/main/scutquant/operators.py):  
为[alpha](https://github.com/HaoningChen/ScutQuant/blob/main/scutquant/alpha.py)模块提供底层计算, 也可单独拿出来构建自定义因子. 该模块旨在方便用户以接近自然语言的方式构造因子  
### [alpha](https://github.com/HaoningChen/ScutQuant/blob/main/scutquant/alpha.py): 
因子模块, 负责批量生成因子，并且复现了qlib的alpha360和alpha158.  
前者是基础价量数据的滞后项, 作除单位处理; 后者主要是技术因子, 复现了qlib的alpha 158并做了微小改动    
### [scutquant](https://github.com/HaoningChen/ScutQuant/blob/main/scutquant/scutquant.py):  
负责处理构建因子外的数据分析和数据处理, 还有一些快速建模函数(支持一键训练线性回归模型(基于sklearn), hybrid模型和lightGBM)   
### [models](https://github.com/HaoningChen/ScutQuant/blob/main/scutquant/models.py):  
使用tensorflow和keras写的模型, 可以方便地一键训练自定义的神经网络. 目前已实现的网络有DNN, LSTM, Bi-LSTM, Attention, CNN和Ensemble    
### [account](https://github.com/HaoningChen/ScutQuant/blob/main/scutquant/account.py):  
账户, 回测时使用虚拟账户进行仿真交易  
### [strategy](https://github.com/HaoningChen/ScutQuant/blob/main/scutquant/strategy.py):  
策略, 可以继承BaseStrategy自己设计策略  
### [signal_generator](https://github.com/HaoningChen/ScutQuant/blob/main/scutquant/signal_generator.py):  
根据模型预测值和策略, 在每一个时间t, 生成下单指令order和交易价格current_price
### [executor](https://github.com/HaoningChen/ScutQuant/blob/main/scutquant/executor.py):  
在account, strategy和signal_generator的基础上进行封装而成的执行器, 负责策略的执行和更新account  
### [report](https://github.com/HaoningChen/ScutQuant/blob/main/scutquant/report.py):  
报告回测结果, 也可用于测试因子和模型预测值的单调性(见group_return_ana()函数)  


### 构建因子和处理数据的alpha模块和scutquant模块依赖operators模块，执行回测的executor模块依赖signal_generator模块（负责根据预测值构建组合，并分配头寸），strategy模块（在signal_generator模块生成信号后，根据策略进一步筛选交易资产）和accout模块（记录基金的交易信息）

## Workflow  
![1](https://user-images.githubusercontent.com/101194077/229791039-833128a9-320b-49be-9848-5b47e2b2f4a8.png)



## 数据处理流程（到执行回测前）  
![2](https://user-images.githubusercontent.com/101194077/209441805-ecee94f8-794a-4431-819f-73f66d182aef.png)
