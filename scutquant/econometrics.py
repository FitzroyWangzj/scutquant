import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from pandas.core.groupby import Grouper
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
from .models import Model


class GARCH(Model):
    def __init__(self, p: int = 1, q: int = 1, mean: str = 'Zero', forecast_horizon: int = 0, *args, **kwargs):
        """
        GARCH模型用于波动率预测
        
        参数:
        p: GARCH模型的p参数，表示条件方差的自回归阶数
        q: GARCH模型的q参数，表示残差平方的移动平均阶数
        mean: 均值模型，可选'Zero'、'Constant'、'AR'等
        forecast_horizon: 预测未来几天的波动率
        """
        super().__init__(*args, **kwargs)
        self.p = p
        self.q = q
        self.mean = mean
        self.forecast_horizon = forecast_horizon
        self.models = {}  # 存储每个金融工具的GARCH模型
        self.model_fits = {}  # 存储每个金融工具的拟合结果
        
    def init_model(self):
        # GARCH模型不需要初始化PyTorch模型
        pass
        
    def fit(self, x_train, returns_col='pct_chg'):
        """
        训练GARCH模型并返回训练集上的条件方差预测
        
        参数:
        x_train: 训练数据，包含收益率列
        returns_col: 收益率列名
        
        返回:
        训练集上的条件方差预测DataFrame
        """
        from arch import arch_model
        
        # 确保x_train是DataFrame
        if not isinstance(x_train, DataFrame):
            raise ValueError("x_train必须是DataFrame")
            
        # 获取所有唯一的金融工具
        instruments = x_train.index.get_level_values(1).unique()
        
        total_loss = 0
        valid_models = 0
        
        # 存储训练集上的条件方差预测
        volatility_predictions = []
        
        # 对每个金融工具单独建模
        for instrument in tqdm(instruments, desc="训练GARCH模型"):
            try:
                instrument_data = x_train.xs(instrument, level=1)
                instrument_data = instrument_data.sort_index()
                returns = instrument_data[returns_col]
                returns = returns.dropna()
                
                if len(returns) <= max(self.p, self.q) + 10:  
                    print(f"警告: {instrument} 的数据点不足，跳过")
                    continue
                
                self.models[instrument] = arch_model(returns, vol='Garch', p=self.p, q=self.q, mean=self.mean)
                self.model_fits[instrument] = self.models[instrument].fit(disp='off')
                
                # 计算损失（负对数似然）
                total_loss += self.model_fits[instrument].loglikelihood
                valid_models += 1
                
                # 获取训练集上的条件方差预测
                conditional_variance = self.model_fits[instrument].conditional_volatility**2
                
                # 创建预测结果
                for date, variance in zip(conditional_variance.index, conditional_variance.values):
                    volatility_predictions.append({
                        'datetime': date,
                        'instrument': instrument,
                        'conditional_variance': variance
                    })
                
            except Exception as e:
                print(f"处理 {instrument} 时出错: {e}")
        
        if valid_models > 0:
            print(f"训练完成，平均对数似然: {total_loss / valid_models}")
        else:
            print("警告: 没有成功训练任何模型")
            
        # 将预测结果转换为DataFrame
        vol_df = DataFrame(volatility_predictions)
        
        # 设置多重索引
        if not vol_df.empty:
            vol_df = vol_df.set_index(['datetime', 'instrument'])
            
        return vol_df
    
    def predict_pandas(self, x: DataFrame, returns_col='pct_chg') -> DataFrame:
        """
        预测每个金融工具的条件方差
        
        参数:
        x: 包含收益率数据的DataFrame
        returns_col: 收益率列名
        
        返回:
        预测的条件方差DataFrame，索引与原始x相同
        """
        if not self.models:
            raise ValueError("模型尚未训练，请先调用fit方法")
            
        instruments = x.index.get_level_values(1).unique()
        volatility_predictions = []

        for instrument in tqdm(instruments, desc="预测条件方差"):
            if instrument not in self.models:
                print(f"警告: {instrument} 没有对应的模型，跳过")
                continue
                
            try:
                # 提取该金融工具的数据
                instrument_data = x.xs(instrument, level=1)
                instrument_data = instrument_data.sort_index()
                
                # 获取模型拟合结果
                model_fit = self.model_fits[instrument]
                
                # 获取条件方差预测
                conditional_variance = model_fit.conditional_volatility**2
                
                # 创建预测结果
                for date, variance in zip(conditional_variance.index, conditional_variance.values):
                    volatility_predictions.append({
                        'datetime': date,
                        'instrument': instrument,
                        'conditional_variance': variance
                    })
                
                # 如果需要预测未来的条件方差
                if self.forecast_horizon > 0:
                    forecast = model_fit.forecast(horizon=self.forecast_horizon)
                    
                    # 获取最后一个预测点
                    last_date = instrument_data.index[-1]
                    
                    # 添加未来预测的条件方差
                    for h in range(1, self.forecast_horizon + 1):
                        future_variance = forecast.variance.iloc[-1, h-1]
                        # 这里假设日期是按天递增的，可以根据实际情况调整
                        future_date = pd.Timestamp(last_date) + pd.Timedelta(days=h)
                        
                        volatility_predictions.append({
                            'datetime': future_date,
                            'instrument': instrument,
                            'conditional_variance': future_variance,
                            'is_forecast': True  # 标记为预测值
                        })
                    
            except Exception as e:
                print(f"预测 {instrument} 时出错: {e}")
        
        # 将预测结果转换为DataFrame
        vol_df = DataFrame(volatility_predictions)
        
        # 设置多重索引
        if not vol_df.empty:
            if 'is_forecast' in vol_df.columns:
                vol_df = vol_df.set_index(['datetime', 'instrument', 'is_forecast'])
            else:
                vol_df = vol_df.set_index(['datetime', 'instrument'])
            
        return vol_df

    def rolling_predict(self, x: DataFrame, returns_col='pct_chg', w=252, step=1, parallel=False, n_jobs=-1) -> DataFrame:
        """
        使用滚动窗口方法预测条件方差
        
        参数:
        x: 包含收益率数据的DataFrame
        returns_col: 收益率列名
        w: 滚动窗口大小，默认为252（约一年的交易日）
        step: 滚动步长，默认为1，可以设置更大的值来减少计算量
        parallel: 是否使用并行计算，默认为False
        n_jobs: 并行计算的作业数，默认为-1（使用所有可用CPU）
        
        返回:
        滚动预测的条件方差DataFrame
        """
        from arch import arch_model
        
        # 确保x是DataFrame
        if not isinstance(x, DataFrame):
            raise ValueError("x必须是DataFrame")
            
        # 获取所有唯一的金融工具
        instruments = x.index.get_level_values(1).unique()
        
        # 存储滚动预测结果
        volatility_predictions = []
        
        # 定义单个金融工具的处理函数
        def process_instrument(instrument):
            instrument_predictions = []
            try:
                # 提取该金融工具的数据
                instrument_data = x.xs(instrument, level=1)
                instrument_data = instrument_data.sort_index()
                
                # 获取收益率数据
                returns = instrument_data[returns_col]
                returns = returns.dropna()
                
                if len(returns) <= max(self.p, self.q) + 10 + w:
                    print(f"警告: {instrument} 的数据点不足，跳过")
                    return []
                
                # 获取所有日期
                dates = returns.index
                
                # 对每个滚动窗口进行预测，使用步长来减少计算量
                for i in range(w, len(dates), step):
                    # 获取当前窗口的数据
                    window_data = returns.iloc[i-w:i]
                    current_date = dates[i]
                    
                    # 创建并拟合GARCH模型
                    model = arch_model(window_data, vol='Garch', p=self.p, q=self.q, mean=self.mean)
                    model_fit = model.fit(disp='off', show_warning=False, options={'maxiter': 100})
                    
                    # 获取最后一天的条件方差
                    conditional_variance = model_fit.conditional_volatility[-1]**2
                    
                    # 存储预测结果
                    instrument_predictions.append({
                        'datetime': current_date,
                        'instrument': instrument,
                        'conditional_variance': conditional_variance
                    })
                    
                    # 如果需要预测未来的条件方差
                    if self.forecast_horizon > 0:
                        forecast = model_fit.forecast(horizon=self.forecast_horizon)
                        
                        # 添加未来预测的条件方差
                        for h in range(1, self.forecast_horizon + 1):
                            future_variance = forecast.variance.iloc[-1, h-1]
                            future_date = pd.Timestamp(current_date) + pd.Timedelta(days=h)
                            
                            instrument_predictions.append({
                                'datetime': future_date,
                                'instrument': instrument,
                                'conditional_variance': future_variance,
                                'is_forecast': True
                            })
                
            except Exception as e:
                print(f"处理 {instrument} 时出错: {e}")
            
            return instrument_predictions
        
        # 使用并行计算处理多个金融工具
        if parallel:
            try:
                from joblib import Parallel, delayed
                
                # 并行处理所有金融工具
                results = Parallel(n_jobs=n_jobs)(
                    delayed(process_instrument)(instrument) for instrument in tqdm(instruments, desc=f"滚动窗口预测 (窗口={w}, 步长={step})")
                )
                
                # 合并结果
                for result in results:
                    volatility_predictions.extend(result)
                    
            except ImportError:
                print("警告: 未安装joblib，无法使用并行计算。请使用 'pip install joblib' 安装。")
                # 回退到串行处理
                for instrument in tqdm(instruments, desc=f"滚动窗口预测 (窗口={w}, 步长={step})"):
                    volatility_predictions.extend(process_instrument(instrument))
        else:
            # 串行处理所有金融工具
            for instrument in tqdm(instruments, desc=f"滚动窗口预测 (窗口={w}, 步长={step})"):
                volatility_predictions.extend(process_instrument(instrument))
        
        # 将预测结果转换为DataFrame
        vol_df = DataFrame(volatility_predictions)
        
        # 设置多重索引
        if not vol_df.empty:
            if 'is_forecast' in vol_df.columns:
                vol_df = vol_df.set_index(['datetime', 'instrument', 'is_forecast'])
            else:
                vol_df = vol_df.set_index(['datetime', 'instrument'])
        
        return vol_df


class EGARCH(Model):
    def __init__(self, p: int = 1, q: int = 1, o: int = 1, mean: str = 'Zero', forecast_horizon: int = 0, *args, **kwargs):
        """
        EGARCH模型用于波动率预测，能够捕捉杠杆效应
        
        参数:
        p: EGARCH模型的p参数，表示条件方差的自回归阶数
        q: EGARCH模型的q参数，表示残差平方的移动平均阶数
        o: EGARCH模型的o参数，表示杠杆效应项的阶数
        mean: 均值模型，可选'Zero'、'Constant'、'AR'等
        forecast_horizon: 预测未来几天的波动率
        """
        super().__init__(*args, **kwargs)
        self.p = p
        self.q = q
        self.o = o
        self.mean = mean
        self.forecast_horizon = forecast_horizon
        self.models = {}  # 存储每个金融工具的EGARCH模型
        self.model_fits = {}  # 存储每个金融工具的拟合结果
        
    def init_model(self):
        # EGARCH模型不需要初始化PyTorch模型
        pass
        
    def fit(self, x_train, returns_col='pct_chg'):
        """
        训练EGARCH模型并返回训练集上的条件方差预测
        
        参数:
        x_train: 训练数据，包含收益率列
        returns_col: 收益率列名
        
        返回:
        训练集上的条件方差预测DataFrame
        """
        from arch import arch_model
        
        # 确保x_train是DataFrame
        if not isinstance(x_train, DataFrame):
            raise ValueError("x_train必须是DataFrame")
            
        # 获取所有唯一的金融工具
        instruments = x_train.index.get_level_values(1).unique()
        
        total_loss = 0
        valid_models = 0
        
        # 存储训练集上的条件方差预测
        volatility_predictions = []
        
        # 对每个金融工具单独建模
        for instrument in tqdm(instruments, desc="训练EGARCH模型"):
            try:
                instrument_data = x_train.xs(instrument, level=1)
                instrument_data = instrument_data.sort_index()
                returns = instrument_data[returns_col]
                returns = returns.dropna()
                
                if len(returns) <= max(self.p, self.q, self.o) + 10:  
                    print(f"警告: {instrument} 的数据点不足，跳过")
                    continue
                
                self.models[instrument] = arch_model(returns, vol='EGARCH', p=self.p, q=self.q, o=self.o, mean=self.mean)
                self.model_fits[instrument] = self.models[instrument].fit(disp='off')
                
                # 计算损失（负对数似然）
                total_loss += self.model_fits[instrument].loglikelihood
                valid_models += 1
                
                # 获取训练集上的条件方差预测
                conditional_variance = self.model_fits[instrument].conditional_volatility**2
                
                # 创建预测结果
                for date, variance in zip(conditional_variance.index, conditional_variance.values):
                    volatility_predictions.append({
                        'datetime': date,
                        'instrument': instrument,
                        'conditional_variance': variance
                    })
                
            except Exception as e:
                print(f"处理 {instrument} 时出错: {e}")
        
        if valid_models > 0:
            print(f"训练完成，平均对数似然: {total_loss / valid_models}")
        else:
            print("警告: 没有成功训练任何模型")
            
        # 将预测结果转换为DataFrame
        vol_df = DataFrame(volatility_predictions)
        
        # 设置多重索引
        if not vol_df.empty:
            vol_df = vol_df.set_index(['datetime', 'instrument'])
            
        return vol_df
    
    def predict_pandas(self, x: DataFrame, returns_col='pct_chg') -> DataFrame:
        """
        预测每个金融工具的条件方差
        
        参数:
        x: 包含收益率数据的DataFrame
        returns_col: 收益率列名
        
        返回:
        预测的条件方差DataFrame，索引与原始x相同
        """
        if not self.models:
            raise ValueError("模型尚未训练，请先调用fit方法")
            
        instruments = x.index.get_level_values(1).unique()
        volatility_predictions = []

        for instrument in tqdm(instruments, desc="预测条件方差"):
            if instrument not in self.models:
                print(f"警告: {instrument} 没有对应的模型，跳过")
                continue
                
            try:
                # 提取该金融工具的数据
                instrument_data = x.xs(instrument, level=1)
                instrument_data = instrument_data.sort_index()
                
                # 获取模型拟合结果
                model_fit = self.model_fits[instrument]
                
                # 获取条件方差预测
                conditional_variance = model_fit.conditional_volatility**2
                
                # 创建预测结果
                for date, variance in zip(conditional_variance.index, conditional_variance.values):
                    volatility_predictions.append({
                        'datetime': date,
                        'instrument': instrument,
                        'conditional_variance': variance
                    })
                
                # 如果需要预测未来的条件方差
                if self.forecast_horizon > 0:
                    forecast = model_fit.forecast(horizon=self.forecast_horizon)
                    
                    # 获取最后一个预测点
                    last_date = instrument_data.index[-1]
                    
                    # 添加未来预测的条件方差
                    for h in range(1, self.forecast_horizon + 1):
                        future_variance = forecast.variance.iloc[-1, h-1]
                        # 这里假设日期是按天递增的，可以根据实际情况调整
                        future_date = pd.Timestamp(last_date) + pd.Timedelta(days=h)
                        
                        volatility_predictions.append({
                            'datetime': future_date,
                            'instrument': instrument,
                            'conditional_variance': future_variance,
                            'is_forecast': True  # 标记为预测值
                        })
                    
            except Exception as e:
                print(f"预测 {instrument} 时出错: {e}")
        
        # 将预测结果转换为DataFrame
        vol_df = DataFrame(volatility_predictions)
        
        # 设置多重索引
        if not vol_df.empty:
            if 'is_forecast' in vol_df.columns:
                vol_df = vol_df.set_index(['datetime', 'instrument', 'is_forecast'])
            else:
                vol_df = vol_df.set_index(['datetime', 'instrument'])
            
        return vol_df

    def rolling_predict(self, x: DataFrame, returns_col='pct_chg', w=252, step=1, parallel=False, n_jobs=-1) -> DataFrame:
        """
        使用滚动窗口方法预测条件方差
        
        参数:
        x: 包含收益率数据的DataFrame
        returns_col: 收益率列名
        w: 滚动窗口大小，默认为252（约一年的交易日）
        step: 滚动步长，默认为1，可以设置更大的值来减少计算量
        parallel: 是否使用并行计算，默认为False
        n_jobs: 并行计算的作业数，默认为-1（使用所有可用CPU）
        
        返回:
        滚动预测的条件方差DataFrame
        """
        from arch import arch_model
        
        # 确保x是DataFrame
        if not isinstance(x, DataFrame):
            raise ValueError("x必须是DataFrame")
            
        # 获取所有唯一的金融工具
        instruments = x.index.get_level_values(1).unique()
        
        # 存储滚动预测结果
        volatility_predictions = []
        
        # 定义单个金融工具的处理函数
        def process_instrument(instrument):
            instrument_predictions = []
            try:
                # 提取该金融工具的数据
                instrument_data = x.xs(instrument, level=1)
                instrument_data = instrument_data.sort_index()
                
                # 获取收益率数据
                returns = instrument_data[returns_col]
                returns = returns.dropna()
                
                if len(returns) <= max(self.p, self.q, self.o) + 10 + w:
                    print(f"警告: {instrument} 的数据点不足，跳过")
                    return []
                
                # 获取所有日期
                dates = returns.index
                
                # 对每个滚动窗口进行预测，使用步长来减少计算量
                for i in range(w, len(dates), step):
                    # 获取当前窗口的数据
                    window_data = returns.iloc[i-w:i]
                    current_date = dates[i]
                    
                    # 创建并拟合EGARCH模型
                    model = arch_model(window_data, vol='EGARCH', p=self.p, q=self.q, o=self.o, mean=self.mean)
                    model_fit = model.fit(disp='off', show_warning=False, options={'maxiter': 100})
                    
                    # 获取最后一天的条件方差
                    conditional_variance = model_fit.conditional_volatility[-1]**2
                    
                    # 存储预测结果
                    instrument_predictions.append({
                        'datetime': current_date,
                        'instrument': instrument,
                        'conditional_variance': conditional_variance
                    })
                    
                    # 如果需要预测未来的条件方差
                    if self.forecast_horizon > 0:
                        forecast = model_fit.forecast(horizon=self.forecast_horizon)
                        
                        # 添加未来预测的条件方差
                        for h in range(1, self.forecast_horizon + 1):
                            future_variance = forecast.variance.iloc[-1, h-1]
                            future_date = pd.Timestamp(current_date) + pd.Timedelta(days=h)
                            
                            instrument_predictions.append({
                                'datetime': future_date,
                                'instrument': instrument,
                                'conditional_variance': future_variance,
                                'is_forecast': True
                            })
                
            except Exception as e:
                print(f"处理 {instrument} 时出错: {e}")
            
            return instrument_predictions
        
        # 使用并行计算处理多个金融工具
        if parallel:
            try:
                from joblib import Parallel, delayed
                
                # 并行处理所有金融工具
                results = Parallel(n_jobs=n_jobs)(
                    delayed(process_instrument)(instrument) for instrument in tqdm(instruments, desc=f"滚动窗口预测 (窗口={w}, 步长={step})")
                )
                
                # 合并结果
                for result in results:
                    volatility_predictions.extend(result)
                    
            except ImportError:
                print("警告: 未安装joblib，无法使用并行计算。请使用 'pip install joblib' 安装。")
                # 回退到串行处理
                for instrument in tqdm(instruments, desc=f"滚动窗口预测 (窗口={w}, 步长={step})"):
                    volatility_predictions.extend(process_instrument(instrument))
        else:
            # 串行处理所有金融工具
            for instrument in tqdm(instruments, desc=f"滚动窗口预测 (窗口={w}, 步长={step})"):
                volatility_predictions.extend(process_instrument(instrument))
        
        # 将预测结果转换为DataFrame
        vol_df = DataFrame(volatility_predictions)
        
        # 设置多重索引
        if not vol_df.empty:
            if 'is_forecast' in vol_df.columns:
                vol_df = vol_df.set_index(['datetime', 'instrument', 'is_forecast'])
            else:
                vol_df = vol_df.set_index(['datetime', 'instrument'])
        
        return vol_df


class HAR(Model):
    def __init__(self, lags=[1, 5, 22], forecast_horizon: int = 0, *args, **kwargs):
        """
        HAR (Heterogeneous Autoregressive) 模型用于波动率预测
        
        参数:
        lags: 滞后期列表，默认为[1, 5, 22]，分别代表日度、周度和月度
        forecast_horizon: 预测未来几天的波动率
        """
        super().__init__(*args, **kwargs)
        self.lags = lags
        self.forecast_horizon = forecast_horizon
        self.models = {}  # 存储每个金融工具的HAR模型参数
        
    def init_model(self):
        # HAR模型不需要初始化PyTorch模型
        pass
    
    def _prepare_har_features(self, returns, volatility_proxy='RV'):
        """
        准备HAR模型的特征
        
        参数:
        returns: 收益率序列
        volatility_proxy: 波动率代理变量，可选'squared_returns'或'absolute_returns'
        
        返回:
        特征矩阵X和目标变量y
        """
        # 计算波动率代理变量
        if volatility_proxy == 'squared_returns':
            vol_proxy = returns**2
        elif volatility_proxy == 'absolute_returns':
            vol_proxy = np.abs(returns)
        else:
            raise ValueError(f"不支持的波动率代理变量: {volatility_proxy}")
        
        # 计算不同滞后期的平均波动率
        X = pd.DataFrame(index=vol_proxy.index)
        
        for lag in self.lags:
            # 计算过去lag天的平均波动率
            X[f'vol_lag_{lag}'] = vol_proxy.rolling(window=lag).mean().shift(1)
        
        # 删除包含NaN的行
        X = X.dropna()
        
        # 准备目标变量
        y = vol_proxy.loc[X.index]
        
        return X, y
    
    def fit(self, x_train, returns_col='pct_chg', volatility_proxy='RV'):
        """
        训练HAR模型并返回训练集上的条件方差预测
        
        参数:
        x_train: 训练数据，包含收益率列
        returns_col: 收益率列名
        volatility_proxy: 波动率代理变量，可选'squared_returns'或'absolute_returns'
        
        返回:
        训练集上的条件方差预测DataFrame
        """
        from sklearn.linear_model import LinearRegression
        
        # 确保x_train是DataFrame
        if not isinstance(x_train, DataFrame):
            raise ValueError("x_train必须是DataFrame")
            
        # 获取所有唯一的金融工具
        instruments = x_train.index.get_level_values(1).unique()
        
        # 存储训练集上的条件方差预测
        volatility_predictions = []
        
        # 对每个金融工具单独建模
        for instrument in tqdm(instruments, desc="训练HAR模型"):
            try:
                # 提取该金融工具的数据
                instrument_data = x_train.xs(instrument, level=1)
                instrument_data = instrument_data.sort_index()
                
                # 获取收益率数据
                returns = instrument_data[returns_col]
                returns = returns.dropna()
                
                # 最大滞后期
                max_lag = max(self.lags)
                
                if len(returns) <= max_lag + 10:  
                    print(f"警告: {instrument} 的数据点不足，跳过")
                    continue
                
                # 准备特征和目标变量
                X, y = self._prepare_har_features(returns, volatility_proxy)
                
                # 训练线性回归模型
                model = LinearRegression()
                model.fit(X, y)
                
                # 存储模型参数
                self.models[instrument] = {
                    'intercept': model.intercept_,
                    'coefficients': model.coef_,
                    'feature_names': X.columns.tolist()
                }
                
                # 预测条件方差
                y_pred = model.predict(X)
                
                # 创建预测结果
                for date, variance in zip(X.index, y_pred):
                    volatility_predictions.append({
                        'datetime': date,
                        'instrument': instrument,
                        'conditional_variance': variance
                    })
                
            except Exception as e:
                print(f"处理 {instrument} 时出错: {e}")
        
        # 将预测结果转换为DataFrame
        vol_df = DataFrame(volatility_predictions)
        
        # 设置多重索引
        if not vol_df.empty:
            vol_df = vol_df.set_index(['datetime', 'instrument'])
            
        return vol_df
    
    def predict_pandas(self, x: DataFrame, returns_col='pct_chg', volatility_proxy='RV') -> DataFrame:
        """
        预测每个金融工具的条件方差
        
        参数:
        x: 包含收益率数据的DataFrame
        returns_col: 收益率列名
        volatility_proxy: 波动率代理变量，可选'squared_returns'或'absolute_returns'
        
        返回:
        预测的条件方差DataFrame
        """
        if not self.models:
            raise ValueError("模型尚未训练，请先调用fit方法")
            
        instruments = x.index.get_level_values(1).unique()
        volatility_predictions = []

        for instrument in tqdm(instruments, desc="预测条件方差"):
            if instrument not in self.models:
                print(f"警告: {instrument} 没有对应的模型，跳过")
                continue
                
            try:
                # 提取该金融工具的数据
                instrument_data = x.xs(instrument, level=1)
                instrument_data = instrument_data.sort_index()
                
                # 获取收益率数据
                returns = instrument_data[returns_col]
                returns = returns.dropna()
                
                # 准备特征
                X, _ = self._prepare_har_features(returns, volatility_proxy)
                
                # 获取模型参数
                model_params = self.models[instrument]
                intercept = model_params['intercept']
                coefficients = model_params['coefficients']
                
                # 预测条件方差
                y_pred = intercept + np.dot(X, coefficients)
                
                # 创建预测结果
                for date, variance in zip(X.index, y_pred):
                    volatility_predictions.append({
                        'datetime': date,
                        'instrument': instrument,
                        'conditional_variance': variance
                    })
                
                # 如果需要预测未来的条件方差
                if self.forecast_horizon > 0:
                    # 获取最后一个预测点的特征
                    last_features = X.iloc[-1].values
                    last_date = X.index[-1]
                    
                    # 预测未来的条件方差
                    for h in range(1, self.forecast_horizon + 1):
                        # 简单地使用最后一个预测作为未来预测
                        # 注意：这是一个简化处理，实际应用中可能需要更复杂的方法
                        future_variance = intercept + np.dot(last_features, coefficients)
                        future_date = pd.Timestamp(last_date) + pd.Timedelta(days=h)
                        
                        volatility_predictions.append({
                            'datetime': future_date,
                            'instrument': instrument,
                            'conditional_variance': future_variance,
                            'is_forecast': True  # 标记为预测值
                        })
                    
            except Exception as e:
                print(f"预测 {instrument} 时出错: {e}")
        
        # 将预测结果转换为DataFrame
        vol_df = DataFrame(volatility_predictions)
        
        # 设置多重索引
        if not vol_df.empty:
            if 'is_forecast' in vol_df.columns:
                vol_df = vol_df.set_index(['datetime', 'instrument', 'is_forecast'])
            else:
                vol_df = vol_df.set_index(['datetime', 'instrument'])
            
        return vol_df

    def rolling_predict(self, x: DataFrame, returns_col='pct_chg', volatility_proxy='RV', 
                        w=252, step=1, parallel=False, n_jobs=-1) -> DataFrame:
        """
        使用滚动窗口方法预测条件方差
        
        参数:
        x: 包含收益率数据的DataFrame
        returns_col: 收益率列名
        volatility_proxy: 波动率代理变量，可选'squared_returns'或'absolute_returns'
        w: 滚动窗口大小，默认为252（约一年的交易日）
        step: 滚动步长，默认为1，可以设置更大的值来减少计算量
        parallel: 是否使用并行计算，默认为False
        n_jobs: 并行计算的作业数，默认为-1（使用所有可用CPU）
        
        返回:
        滚动预测的条件方差DataFrame
        """
        from sklearn.linear_model import LinearRegression
        
        # 确保x是DataFrame
        if not isinstance(x, DataFrame):
            raise ValueError("x必须是DataFrame")
            
        # 获取所有唯一的金融工具
        instruments = x.index.get_level_values(1).unique()
        
        # 存储滚动预测结果
        volatility_predictions = []
        
        # 定义单个金融工具的处理函数
        def process_instrument(instrument):
            instrument_predictions = []
            try:
                # 提取该金融工具的数据
                instrument_data = x.xs(instrument, level=1)
                instrument_data = instrument_data.sort_index()
                
                # 获取收益率数据
                returns = instrument_data[returns_col]
                returns = returns.dropna()
                
                # 最大滞后期
                max_lag = max(self.lags)
                
                if len(returns) <= max_lag + 10 + w:
                    print(f"警告: {instrument} 的数据点不足，跳过")
                    return []
                
                # 获取所有日期
                all_dates = returns.index
                
                # 对每个滚动窗口进行预测，使用步长来减少计算量
                for i in range(w + max_lag, len(all_dates), step):
                    # 获取当前窗口的数据
                    window_returns = returns.iloc[i-w:i]
                    current_date = all_dates[i]
                    
                    # 准备特征和目标变量
                    X, y = self._prepare_har_features(window_returns, volatility_proxy)
                    
                    # 只使用窗口内的数据
                    X = X.iloc[-w:]
                    y = y.iloc[-w:]
                    
                    # 训练线性回归模型
                    model = LinearRegression()
                    model.fit(X, y)
                    
                    # 预测最后一天的条件方差
                    last_features = X.iloc[-1].values.reshape(1, -1)
                    conditional_variance = model.predict(last_features)[0]
                    
                    # 存储预测结果
                    instrument_predictions.append({
                        'datetime': current_date,
                        'instrument': instrument,
                        'conditional_variance': conditional_variance
                    })
                    
                    # 如果需要预测未来的条件方差
                    if self.forecast_horizon > 0:
                        # 简单地使用最后一个预测作为未来预测
                        for h in range(1, self.forecast_horizon + 1):
                            future_variance = conditional_variance  # 简化处理
                            future_date = pd.Timestamp(current_date) + pd.Timedelta(days=h)
                            
                            instrument_predictions.append({
                                'datetime': future_date,
                                'instrument': instrument,
                                'conditional_variance': future_variance,
                                'is_forecast': True
                            })
                
            except Exception as e:
                print(f"处理 {instrument} 时出错: {e}")
            
            return instrument_predictions
        
        # 使用并行计算处理多个金融工具
        if parallel:
            try:
                from joblib import Parallel, delayed
                
                # 并行处理所有金融工具
                results = Parallel(n_jobs=n_jobs)(
                    delayed(process_instrument)(instrument) for instrument in tqdm(instruments, desc=f"滚动窗口预测 (窗口={w}, 步长={step})")
                )
                
                # 合并结果
                for result in results:
                    volatility_predictions.extend(result)
                    
            except ImportError:
                print("警告: 未安装joblib，无法使用并行计算。请使用 'pip install joblib' 安装。")
                # 回退到串行处理
                for instrument in tqdm(instruments, desc=f"滚动窗口预测 (窗口={w}, 步长={step})"):
                    volatility_predictions.extend(process_instrument(instrument))
        else:
            # 串行处理所有金融工具
            for instrument in tqdm(instruments, desc=f"滚动窗口预测 (窗口={w}, 步长={step})"):
                volatility_predictions.extend(process_instrument(instrument))
        
        # 将预测结果转换为DataFrame
        vol_df = DataFrame(volatility_predictions)
        
        # 设置多重索引
        if not vol_df.empty:
            if 'is_forecast' in vol_df.columns:
                vol_df = vol_df.set_index(['datetime', 'instrument', 'is_forecast'])
            else:
                vol_df = vol_df.set_index(['datetime', 'instrument'])
        
        return vol_df



















