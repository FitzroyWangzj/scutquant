import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def sharpe_ratio(ret: pd.Series | pd.DataFrame, rf: float = 0.03, freq: float = 1.0) -> float:
    """
    夏普比率（事后夏普比率，使用实际收益计算）

    :param ret: pd.Series or pd.DataFrame, 收益率曲线
    :param rf: float, 无风险利率
    :param freq: float, 时间频率, 以年为单位则为1, 天为单位则为365, 其它类推
    :return: float, 对应时间频率的夏普比率
    """
    ret_copy = pd.Series(ret)
    rf = (1 + rf) ** (1 / freq) - 1
    return (ret_copy.mean() - rf) / ret_copy.std()


def sortino_ratio(ret: pd.Series | pd.DataFrame, benchmark: pd.Series | pd.DataFrame) -> float:
    """
    索提诺比率

    :param ret: pd.Series or pd.DataFrame, 收益率曲线
    :param benchmark: pd.Series or pd.DataFrame, 基准回报率曲线
    :return: float, 对应时间频率的夏普比率
    """
    ret = pd.Series(ret)
    benchmark_copy = pd.Series(benchmark)
    sd = ret[ret.values < benchmark_copy.values].std()
    return (ret.mean() - benchmark_copy.mean()) / sd


def information_ratio(ret: pd.Series | pd.DataFrame, benchmark: pd.Series | pd.DataFrame) -> float:
    """
    信息比率（与夏普比率的区别是使用指数收益作为对照的标准）

    :param ret: pd.Series or pd.DataFrame, 收益率曲线
    :param benchmark: pd.Series or pd.DataFrame, 基准回报率曲线
    :return: float
    """
    ret_copy = pd.Series(ret)
    benchmark_copy = pd.Series(benchmark)
    return (ret_copy.mean() - benchmark_copy.mean()) / (ret_copy - benchmark_copy).std()


def calc_drawdown(data: pd.Series) -> pd.Series:
    """
    :param data: 累计收益率序列
    :return: 从开始到目前的回撤
    """
    if data.values[0] < 0.5:  # 如果是从0开始则需要+1然后计算, 从1开始则可以直接计算
        data_ = data + 1
        return (data_ - data_.cummax()) / data_.cummax()
    else:
        return (data - data.cummax()) / data.cummax()


def annualized_return(data: pd.Series, freq: float = 1) -> float:
    # (1 + total_ret) ** (1/years) - 1
    return (1 + data.values[-1]) ** (252 / (len(data) * freq)) - 1


def annualized_volatility(data: pd.Series, freq: float = 1) -> float:
    # ret.std()*(x **0.5), x为一年有多少tick
    return data.std() * ((252 / len(data) / freq) ** 0.5)


def plot(data, label, title: str = None, xlabel: str = None, ylabel: str = None, figsize=None,
         mode: str = "plot") -> None:
    """
    :param data: 需要绘制的数据
    :param label: 数据标签
    :param title: 标题
    :param xlabel: x轴的名字
    :param ylabel: y轴的名字
    :param figsize: 图片大小
    :param mode: 画图模式，有折线图”plot“和柱状图"bar"两种可供选择
    :return:
    """
    if figsize is not None:
        plt.figure(figsize=figsize)
    # plt.clf()
    if mode == "plot":
        if len(data) > 10:
            cmap = plt.get_cmap('tab20')
            for d in range(len(data)):
                plt.plot(data[d], label=label[d], color=cmap(d))
                plt.xticks(rotation=45)
        else:
            for d in range(len(data)):
                plt.plot(data[d], label=label[d])
                plt.xticks(rotation=45)
    elif mode == "bar":
        bar = plt.bar(label, data, label="value")
        plt.bar_label(bar, label_type='edge')
        plt.xticks(rotation=45)
    else:
        raise ValueError("We don't support this mode: " + mode)
    if title is not None:
        plt.title(title)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    plt.legend()
    plt.show()


def accuracy(pred: pd.Series, y: pd.Series, sign: str = ">=") -> float:
    """
    eg:
    y = pd.Series([-1, -1, 2, 3])
    y_hat = pd.Series([0, -2, 0, 2])
    data = y * y_hat
    label = 0
    sign = [">" for _ in range(len(data))]
    acc = accuracy(data, label, sign)  # A prediction is accurate if data[i] > label

    :param pred: pd.Series, 预测值
    :param y: pd.Series, 目标值
    :param sign: list, 运算符号, 长度与data相同
    :return: float, accuracy of prediction
    """
    data = pred * y
    # print(data)
    data_true = eval("data[data" + sign + "y]")
    return len(data_true) / len(data)


def report_all(user_account, benchmark, show_raw_value: bool = False, rf: float = 0.03, freq: float = 1, time=None,
               figsize: tuple = (10, 6)) -> None:
    """

    :param user_account: account类
    :param benchmark: account类
    :param show_raw_value: 显示原始市值（具体金额）
    :param rf: 显示无风险利率
    :param freq: 频率, 日频为1，月频为30，其它类推
    :param time: 显示时间轴
    :param figsize: 图片大小
    :return:
    """
    if time is not None:
        time = pd.to_datetime(time, format='%Y-%m-%d')

    acc_val, ben_val = user_account.val_hist, benchmark.val_hist  # with cost
    init_val_acc = acc_val[0]
    init_val_ben = ben_val[0]

    acc_ret = []
    ben_ret = []
    days = 0
    for i in range(len(acc_val)):
        acc_ret.append(acc_val[i] / init_val_acc - 1)  # 当前净值相对于初始值的收益率
        ben_ret.append(ben_val[i] / init_val_ben - 1)
    excess_ret = []

    for i in range(len(acc_ret)):
        excess_ret.append(acc_ret[i] - ben_ret[i])
        if acc_ret[i] - ben_ret[i] > 0:
            days += 1
    days /= len(acc_ret)

    acc_ret = pd.Series(acc_ret, name="acc_ret", index=time)  # 累计收益率
    ben_ret = pd.Series(ben_ret, name="ben_ret", index=time)  # benchmark的累计收益率
    excess_ret = pd.Series(excess_ret, name="excess_ret", index=time)

    acc_ret = pd.concat([pd.Series([0], index=[acc_ret.index[0] - pd.DateOffset(days=1)]),
                         acc_ret], axis=0)
    ben_ret = pd.concat([pd.Series([0], index=[ben_ret.index[0] - pd.DateOffset(days=1)]),
                         ben_ret], axis=0)
    excess_ret = pd.concat([pd.Series([0], index=[excess_ret.index[0] - pd.DateOffset(days=1)]),
                            excess_ret], axis=0)

    acc_dd = calc_drawdown(acc_ret)
    ben_dd = calc_drawdown(ben_ret)
    excess_dd = calc_drawdown(excess_ret)

    ann_return = annualized_return(acc_ret, freq=freq)
    ann_std = annualized_volatility(acc_ret, freq=freq)
    ben_ann_return = annualized_return(ben_ret, freq=freq)
    ben_ann_std = annualized_volatility(ben_ret, freq=freq)

    beta = acc_ret.cov(ben_ret) / ben_ret.var()
    alpha = acc_ret.mean() - beta * ben_ret.mean()
    epsilon = (acc_ret - beta * ben_ret - alpha).std()

    sharpe = sharpe_ratio(acc_ret, rf=rf, freq=freq * 365)
    sortino = sortino_ratio(acc_ret, ben_ret)
    inf_ratio = information_ratio(acc_ret, ben_ret)

    print('Annualized Return:', ann_return)  # (1 + total_ret) ** (1/years) - 1
    # print("years:", 252 / len(ret))
    print('Annualized Volatility:', ann_std)  # ret.std()*(x **0.5), x为一年有多少tick
    print('Annualized Return(Benchmark):', ben_ann_return)
    print('Annualized Volatility(Benchmark):', ben_ann_std, '\n')
    print('Cumulative Rate of Return:', acc_ret[-1])
    print('Cumulative Rate of Return(Benchmark):', ben_ret[-1])
    print('Cumulative Excess Rate of Return:', excess_ret[-1], '\n')
    print('Max Drawdown:', acc_dd.min())
    print('Max Drawdown(Benchmark):', ben_dd.min())
    print('Max Drawdown(Excess Return):', calc_drawdown(pd.Series(excess_ret) + 1).min(), '\n')
    print('Sharpe Ratio:', sharpe)
    print('Sortino Ratio:', sortino)
    print('Information Ratio:', inf_ratio, '\n')
    print('Beta:', beta)
    print("Alpha:", alpha)
    print("Epsilon:", epsilon)
    print('Profitable Days(%):', days)

    if show_raw_value:
        acc_val = pd.Series(acc_val, name="acc_val", index=time)
        ben_val = pd.Series(ben_val, name="ben_val", index=time)
        plt.figure(figsize=(10, 6))
        plt.plot(acc_val, label="return", color="red")
        plt.plot(ben_val, label="benchmark", color="blue")
        plt.plot(acc_val - ben_val, label="excess_return", color="orange")
        plt.legend()
        plt.show()
    else:
        plt.figure(figsize=(10, 6))
        plt.plot(acc_ret, label="return", color="red")
        plt.plot(ben_ret, label="benchmark", color="blue")
        plt.plot(excess_ret, label="excess_return", color="orange")
        plt.legend()
        plt.title("Returns")
        plt.show()

    plt.clf()
    plt.figure(figsize=(10, 6))
    plt.plot(acc_dd, label="drawdown")
    plt.plot(excess_dd, label="excess_return_drawdown")
    plt.legend()
    plt.title("Drawdown")
    plt.show()

    risk = pd.DataFrame({'risk': user_account.risk_curve}, index=time)
    plot([risk], label=['risk_degree'], title='Risk Degree', ylabel='value', figsize=figsize)

    risk = pd.DataFrame({'turnover': user_account.turnover}, index=time)
    plot([risk], label=['turnover'], title='Turnover', figsize=figsize)


def group_return_ana(pred: pd.DataFrame | pd.Series, y_true: pd.Series, n: int = 10, groupby: str = "time",
                     figsize: tuple = (10, 6), to_df = False, compound = False) -> None:
    """
    因子对股票是否有良好的区分度, 若有, 则应出现明显的分层效应(即单调性)
    此处的收益为因子收益率，非真实收益率

    :param pred: pd.DataFrame or pd.Series, 预测值
    :param y_true: pd.Series, 真实的收益率
    :param n: int, 分组数量(均匀地分成n组)
    :param groupby: str, groupby的索引
    :param figsize: 图片大小
    :params to_df: 输出df
    :params compound: 是否复利
    :return:
    """
    y_true.name = "label"
    y_true.index.names = pred.index.names

    if len(pred) > len(y_true):
        pred = pred[pred.index.isin(y_true.index)]
    else:
        y_true = y_true[y_true.index.isin(pred.index)]

    predict = pd.concat([pred, y_true], axis=1)
    predict = predict.sort_values("predict", ascending=False)
    acc = report.accuracy(predict["predict"], predict["label"], sign=">=")
    print('Accuracy of Prediction:', acc)
    t_df = pd.DataFrame(
        {
            "Group%d"
            % (i + 1): predict.groupby(level=groupby)["label"].apply(
                lambda x: x[len(x) // n * i: len(x) // n * (i + 1)].mean()  # 第 len(x) // n * i 行到 len(x) //n * (i+1) 行
            )
            for i in range(n)
        }
    )
    # 多空组合(即做多排名靠前的股票，做空排名靠后的股票)
    t_df["long-short"] = t_df["Group1"] - t_df["Group%d" % n]

    # Long-Average
    t_df["long-average"] = t_df["Group1"] - predict.groupby(level=groupby)["label"].mean()
    # t_df.index = np.arange(len(t_df.index))
    # print(t_df.head(5))
    cols = t_df.columns
    data = []
    label = []
    win_rate = []
    mean_ret = []
    for c in cols:
        if compound:
            data.append((t_df[c]+1).cumprod()-1)
            label.append(c)
            win_rate.append(round(len(t_df[t_df[c] >= 0]) / len(t_df), 4))
            mean_ret.append(round((t_df[c]+1).cumprod().values[-1]  ** (252/len(t_df)) - 1, 4))
        else:
            data.append(t_df[c].cumsum())
            label.append(c)
            win_rate.append(round(len(t_df[t_df[c] >= 0]) / len(t_df), 4))
            mean_ret.append(round(t_df[c].cumsum().values[-1] / len(t_df) * 252, 4))
    report.plot(data, label, title='Grouped Return', xlabel='date', ylabel='value', figsize=figsize)
    report.plot(win_rate, label=cols, title="Win Rate of Each Group", mode="bar", figsize=figsize)
    report.plot(mean_ret, label=cols, title="Annualized Return of Each Group(%)", mode="bar", figsize=figsize)

    if to_df:
        if compound:
            return (t_df[c]+1).cumprod()
        else:
            return t_df.cumsum()+1


def calc_features(feature: pd.Series):
    """
    对于一个有多重索引[("datetime", "instrument)]的pd.Series, 分别计算:
    (1) 每个datetime的截面上均值和标准差
    (2) 每个datetime的unique_value占比(用于判断特征在当日是数值特征还是类别特征)
    (3) 每个datetime的autocorr(lag=1)
    """
    mean = feature.groupby(level=0).mean()
    std = feature.groupby(level=0).std()
    unique = feature.groupby(level=0).apply(lambda x: len(x.unique()) / len(x))
    autocorr = feature.groupby(level=0).apply(lambda x: x.autocorr(lag=1))
    return mean, std, unique, autocorr


def single_factor_ana(feature: pd.Series):
    """
    输出4张图, 从上到下分别是:
    (1) feature的总体分布
    (2) feature在截面上的均值和标准差
    (3) feature每日的unique_value占比
    (4) feature的autocorr(lag=1)
    """
    mean, std, unique, autocorr = calc_features(feature=feature)

    fig, (ax0, ax1, ax2, ax3) = plt.subplots(nrows=4, ncols=1,figsize=(10,10))
    ax0.hist(feature, bins=100)
    ax0.set_ylabel("count")

    ax1.plot(mean, label="mean")
    ax1.set_ylabel("mean")
    ax1.legend()
    ax11 = ax1.twinx()
    ax11.plot(std, color="orange", label="std")
    ax11.set_ylabel("std")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax11.get_legend_handles_labels()
    ax1.legend().set_visible(False)
    ax11.legend(h1 + h2, l1 + l2)

    ax2.plot(unique)
    ax2.set_ylabel("unique")

    ax3.plot(autocorr)
    ax3.set_ylabel("autocorr")
    ax0.set_title(feature.name)

    plt.subplots_adjust(hspace=0.5)
    plt.show()

def calc_autocorr(feature: pd.Series, lag: int = 1) -> pd.Series:
    """
    计算每个金融工具的自相关系数（使用groupby优化版本）
    
    参数:
    feature: 有多重索引[("datetime", "instrument")]的pd.Series
    lag: 自相关的滞后期数，默认为1
    
    返回:
    pd.Series: 索引为instrument，值为该instrument的自相关系数
    """
    # 确保输入是Series且有多重索引
    if not isinstance(feature, pd.Series) or len(feature.index.names) != 2:
        raise ValueError("输入必须是具有多重索引[('datetime', 'instrument')]的pd.Series")
    
    # 使用groupby直接计算每个instrument的自相关系数
    autocorr_series = feature.groupby(level=1).apply(lambda x: x.sort_index().autocorr(lag=lag))
    autocorr_series.name = f'Autocorrelation (lag={lag})'
    
    # 绘制每个金融工具的自相关系数
    plt.figure(figsize=(12, 6))
    autocorr_series.plot(kind='bar')
    plt.title(f'{feature.name} - Autocorrelation of each instrument (lag={lag})')
    plt.xlabel('Instrument')
    plt.ylabel(f'Autocorrelation (lag={lag})')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()
    
    # 绘制自相关系数的分布直方图
    plt.figure(figsize=(10, 6))
    plt.hist(autocorr_series.values, bins=20)
    plt.title(f'{feature.name} - Distribution of autocorrelation (lag={lag})')
    plt.xlabel(f'Autocorrelation (lag={lag})')
    plt.ylabel('Frequency')
    plt.axvline(x=0, color='r', linestyle='--')
    plt.tight_layout()
    plt.show()
    
    return autocorr_series


def calc_fitness_metrics(pred: pd.Series, y: pd.Series):
    # 计算MAE
    mae = np.mean(np.abs(pred - y))
    
    # 计算MSE
    mse = np.mean((pred - y) ** 2)
    
    # 计算RMSE
    rmse = np.sqrt(mse)
    
    # 计算SMAPE
    smape = np.mean(2 * np.abs(pred - y) / (np.abs(pred) + np.abs(y)))
    
    # 计算RMSPE
    rmspe = np.sqrt(np.mean((np.abs(pred - y) / np.abs(y)) ** 2))
    
    # 计算R方
    ss_res = np.sum((pred - y) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    
    # 将结果以字典形式返回
    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'SMAPE': smape,
        'RMSPE': rmspe,
        'R_squared': r_squared
    }
