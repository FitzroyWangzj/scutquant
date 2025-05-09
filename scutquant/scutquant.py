import datetime
from seaborn import kdeplot
import matplotlib.pyplot as plt
import xgboost
import lightgbm as lgb
from sklearn import linear_model
import pickle
import random
import warnings
from .report import single_factor_ana
from .operators import *

warnings.filterwarnings("ignore")
random.seed(2046)


def join_data(data: pd.DataFrame, data_join: pd.DataFrame, on: str = 'datetime', col: list = None, index: list = None) \
        -> pd.DataFrame:
    """
    将序列数据(例如宏观的利率数据)按时间整合到面板数据中(例如沪深300成分)
    example:

    df_train = scutquant.join_data(df_train, series_train, col=['index_return', 'rf'])
    df_test = scutquant.join_data(df_test, series_test, col=['index_return', 'rf'])
    df = pd.concat([df_train, df_test], axis=0)

    :param data: pd.Series or pd.DataFrame, 股票数据(面板数据)
    :param data_join: pd.Series or pd.DataFrame, 要合并的序列数据
    :param on: str, 表示时间(或者其它)的列(两个数据集同时拥有)
    :param col: list, 被合并数据的列名(必须在data_join中存在)
    :param index: list, 合并后数据的index
    """
    if col is None:
        col = data_join.columns
    if index is None:
        index = data.index.names
    result = pd.merge(data.reset_index(), data_join[col].reset_index(), on=on, how="left")
    return result.set_index(index)


def vlookup(df1: pd.DataFrame, df2: pd.DataFrame, lookup_key: str, date: str = "datetime",
            raw: bool = False) -> pd.DataFrame:
    """
    通过给定df1的lookupkey, 在df2中查找符合条件的值并合并到df1中. 可用于处理另类数据、基本面数据与量价数据的合并

    Example:

    假设我们有两个datetime和instrument都不完全匹配的DataFrame, 一个是量价数据集df, 另一个是新闻数据集news, 它们的索引都是
    [(datetime, instrument)], 现在使用vlookup将news的计算结果按照instrument模糊匹配, 并按照datetime合并到df上:

    news_volume = news.groupby(["datetime", "instrument"])["title].count().to_frame(name="snt_volume")
    merge_df = vlookup(df, news_volume, lookup_key="instrument")
    """

    def match(x):
        unique = df2[lookup_key].unique()
        val = np.nan
        for u in unique:
            if u in x:
                val = u
                break
        return val

    df1.reset_index(inplace=True)
    df2.reset_index(inplace=True)
    original_keys = df1[lookup_key].copy()
    df1[lookup_key] = df1[lookup_key].apply(match)
    merged = pd.merge(df1, df2, on=[date, lookup_key], how="outer")
    if raw:
        merged["key"] = df1[lookup_key]
        merged[lookup_key] = original_keys
        merge = merged.set_index([date, lookup_key, "key"]).sort_index()
        merge = merge[~merge.index.get_level_values(1).isnull()]
        return merge[~merge.index.get_level_values(2).isnull()]
    else:
        merged[lookup_key] = original_keys
        merge = merged.set_index([datetime, lookup_key]).sort_index()
        return merge[~merge.index.get_level_values(1).isnull()]


####################################################
# 特征工程
####################################################
def price2ret(price: pd.DataFrame | pd.Series, shift1: int = -1, shift2: int = -2, groupby: str = None,
              fill: bool = False) -> pd.Series:
    """
    return_rate = price_shift2 / price_shift1 - 1

    :param price: pd.DataFrame
    :param shift1: int, the value shift as denominator
    :param shift2: int, the value shift as numerator
    :param groupby: str
    :param fill: bool
    :return: pd.Series
    """
    if groupby is None:
        ret = price.shift(shift2) / price.shift(shift1).fillna(price.mean) - 1
    else:
        shift_1 = price.groupby([groupby]).shift(shift1)
        shift_2 = price.groupby([groupby]).shift(shift2)
        ret = shift_2 / shift_1 - 1
    if fill:
        ret.fillna(0, inplace=True)
    return ret


def make_pca(X: pd.DataFrame | pd.Series) -> dict:
    from sklearn.decomposition import PCA
    index = X.index
    pca = PCA()
    X_pca = pca.fit_transform(X)
    component_names = [f"PC{i + 1}" for i in range(X_pca.shape[1])]
    X_pca = pd.DataFrame(X_pca, columns=component_names, index=index)
    loadings = pd.DataFrame(
        pca.components_.T,  # transpose the matrix of loadings
        columns=component_names,  # so the columns are the principal components
        index=X.columns,  # and the rows are the original features
    )
    result = {
        "pca": pca,
        "loadings": loadings,
        "X_pca": X_pca
    }
    return result


def plot_pca_variance(pca):
    # Create figure
    fig, axs = plt.subplots(1, 2)
    n = pca.n_components_
    grid = np.arange(1, n + 1)
    # Explained variance
    evr = pca.explained_variance_ratio_
    axs[0].bar(grid, evr)
    axs[0].set(
        xlabel="Component", title="% Explained Variance", ylim=(0.0, 1.0)
    )
    # Cumulative Variance
    cv = np.cumsum(evr)
    axs[1].plot(np.r_[0, grid], np.r_[0, cv], "o-")
    axs[1].set(
        xlabel="Component", title="% Cumulative Variance", ylim=(0.0, 1.0)
    )
    # Set up figure
    fig.set(figwidth=8, dpi=100)
    return axs


def make_mi_scores(X: pd.DataFrame | pd.Series, y: pd.DataFrame | pd.Series) -> pd.Series:
    """
    :param X: pd.DataFrame, 输入的特征
    :param y: pd.DataFrame or pd.Series, 输入的目标值
    :return: pd.Series, index为特征名，value为mutual information
    """
    from sklearn.feature_selection import mutual_info_regression
    # Label encoding for categoricals
    for colname in X.select_dtypes("object"):
        X[colname], _ = X[colname].factorize()
    # All discrete features should now have integer dtypes (double-check this before using MI!)
    discrete_features = X.dtypes == int
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores


def make_r_scores(X: pd.DataFrame | pd.Series, y: pd.DataFrame | pd.Series) -> pd.Series:
    """
    :param X: pd.DataFrame or pd.Series, 特征值
    :param y: pd.DataFrame or pd.Series, 目标值
    :return: pd.Series, index为特征名, value为相关系数
    """
    r: list[float] = []
    cols = X.columns
    for c in cols:
        r.append(pearson_corr(X[c], y))
    result = pd.Series(r, index=cols, name='R Scores').sort_values(ascending=False)
    return result


def show_dist(X: pd.Series | pd.DataFrame) -> None:
    """
    画出数据分布(密度)
    """
    kdeplot(X, shade=True)
    plt.show()


####################################################
# 数据清洗
####################################################
def align(x: pd.Series | pd.DataFrame, y: pd.Series | pd.DataFrame) \
        -> tuple[pd.DataFrame | pd.Series, pd.DataFrame | pd.Series]:
    """
    使x和y有相同的索引
    """
    x = x[x.index.isin(y.index)]
    y = y[y.index.isin(x.index)]
    x = x[x.index.isin(y.index)]
    return x, y


def percentage_missing(X: pd.Series | pd.DataFrame) -> float:
    percent_missing: float = 100 * ((X.isnull().sum()).sum() / np.product(X.shape))
    return percent_missing


def clean(X: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    X.dropna(axis=1, how='all', inplace=True)
    X.fillna(method='ffill', inplace=True)
    X.dropna(axis=0, inplace=True)
    return X


def down_sample(data: pd.Series, threshold: float = 0.5) -> pd.Series:
    """
    对于一个具有多重索引的pd.Series, 去除其unique_value占比低于threshold的天

    :param data: pd.DataFrame, 输入的数据
    :param threshold: str, 需要降采样的列名
    :return: pd.DataFrame, 降采样后的数据集
    """
    unique_val_pct: pd.Series = data.groupby(level=0).apply(lambda x: len(x.unique()) / len(x))
    unexpected_days = unique_val_pct[unique_val_pct < threshold].index
    return data[~data.index.get_level_values(0).isin(unexpected_days)]


def bootstrap(X: pd.DataFrame, col: str, val: int = 0, windows: int = 5, n: float = 0.35) -> pd.DataFrame:
    """
    :param X: pd.DataFrame，输入的数据
    :param col: str, 需要升采样的列名
    :param val: 需要升采样的样本的值
    :param windows: int, 移动平均窗口，用来构建新样本
    :param n: float, 升采样比例，0~1
    :return: pd.DataFrame，扩充后的数据集
    """
    X_tar = X[X[col] == val]
    n_boot_drop = int(len(X_tar) * (1 - n))
    X_sample = pd.DataFrame(columns=X.columns, index=X_tar.index)
    for c in X_tar.columns:
        X_sample[c] = X_tar[c].rolling(window=windows, center=True, min_periods=int(0.5 * windows)).mean()
    choice = np.random.choice(X_sample.index, n_boot_drop, replace=False)
    X_sample = X_sample.drop(choice, axis=0)
    # print(X_sample)
    X = pd.concat((X, X_sample))
    return X

def adjust_data_for_rnn(X_train: pd.DataFrame, X_valid: pd.DataFrame, X_test: pd.DataFrame, y_valid: pd.DataFrame, timestep: int):
    """
    rnn需要t-timestep的多步数据进行预测
    """
    timestep -= 1

    # 补充 X_valid 数据
    unique_datetimes_train = X_train.index.get_level_values(0).unique()
    last_datetimes_train = unique_datetimes_train[-timestep:]  # 获取 X_train 中最后 timestep 个 datetime

    # 从 X_train 中提取最后 timestep 个时间步的数据
    last_timestep_data_train = X_train.loc[(last_datetimes_train, slice(None)), :]

    # 将提取的数据拼接到 X_valid 的开头
    X_valid = pd.concat([last_timestep_data_train, X_valid], axis=0)

    # 补充 X_test 数据
    unique_datetimes_valid = X_valid.index.get_level_values(0).unique()
    last_datetimes_valid = unique_datetimes_valid[-timestep:]  # 获取 X_valid 中最后 timestep 个 datetime

    # 从 X_valid 中提取最后 timestep 个时间步的数据
    last_timestep_data_valid = X_valid.loc[(last_datetimes_valid, slice(None)), :]

    # 将提取的数据拼接到 X_test 的开头
    X_test = pd.concat([last_timestep_data_valid, X_test], axis=0)

    # 重新排序索引（可选）
    X_valid = X_valid.sort_index()
    X_test = X_test.sort_index()

    # 剔除 X_valid 中有但 y_valid 中没有的 instrument
    valid_instruments = y_valid.index.get_level_values(1).unique()  # 获取 y_valid 中存在的 instrument
    X_valid = X_valid[X_valid.index.get_level_values(1).isin(valid_instruments)]

    return X_train, X_valid, X_test


####################################################
# 拆分数据集
####################################################
def split_by_date(X: pd.DataFrame | pd.Series, train_start_date: str, train_end_date: str, valid_start_date: str,
                  valid_end_date: str) -> tuple[pd.DataFrame | pd.Series, pd.DataFrame | pd.Series]:
    """
    :param X: pd.DataFrame
    :param train_start_date: str, 训练集的第一天, 例如“2020-12-28”
    :param train_end_date: str, 训练集最后一天
    :param valid_start_date: str, 验证集第一天, 例如"2022-12-28"
    :param valid_end_date: str, 验证集最后一天
    :return: pd.DataFrame, pd.DataFrame
    """
    X_train = X[X.index.get_level_values(0) <= train_end_date]
    X_train = X_train[X_train.index.get_level_values(0) >= train_start_date]
    X_valid = X[X.index.get_level_values(0) <= valid_end_date]
    X_valid = X_valid[X_valid.index.get_level_values(0) >= valid_start_date]
    return X_train, X_valid


def split(X: pd.DataFrame | pd.Series, params: dict = None) -> \
        tuple[pd.DataFrame | pd.Series, pd.DataFrame | pd.Series]:
    """
    相当于sklearn的train_test_split
    :param X: pd.DataFrame
    :param params: dict, 键名包括 "train", "valid", 值为比例
    :return: pd.DataFrame, pd.DataFrame
    """
    if params is None:
        params = {
            "train": 0.7,
            "valid": 0.3,
        }
    idx = X.index
    lis = [_ for _ in range(len(idx))]
    sample = random.sample(lis, int(len(lis) * params["valid"] + 0.5))
    idx_sample = idx[sample]
    X_valid = X[X.index.isin(idx_sample)]
    X_train = X[~X.index.isin(idx_sample)]
    return X_train, X_valid


def group_split(X: pd.DataFrame | pd.Series, params: dict = None) -> \
        tuple[pd.DataFrame | pd.Series, pd.DataFrame | pd.Series]:
    """
    以当天的所有股票为整体, 随机按比例拆出若干天作为训练集和验证集
    :param X: pd.DataFrame
    :param params: dict, 键名包括 "train", "valid", 值为比例
    :return: pd.DataFrame, pd.DataFrame
    """
    if params is None:
        params = {
            "train": 0.7,
            "valid": 0.3,
        }
    time = X.index.get_level_values(0).unique().values
    lis = [_ for _ in range(len(time))]
    sample = random.sample(lis, int(len(lis) * params["valid"] + 0.5))
    X_valid = X[X.index.get_level_values(0).isin(time[sample])]
    X_train = X[~X.index.isin(X_valid.index)]
    return X_train, X_valid


def split_data_by_date(data: pd.DataFrame | pd.Series, kwargs: dict) -> \
        tuple[pd.DataFrame | pd.Series, pd.DataFrame | pd.Series, pd.DataFrame | pd.Series]:
    """
    按照日期拆出(整段)的测试集, 然后剩下的数据按照参数"split_method"和"split_kwargs"拆除训练集和验证集
    :param data: pd.DataFrame
    :param kwargs: dict, test_start_date必填, 其它选填. 当没指定test_end_date时, 默认截取到最后一天
    :return: pd.DataFrame
    """
    split_method = "split" if "split_method" not in kwargs.keys() else kwargs["split_method"]
    split_kwargs = None if "split_kwargs" not in kwargs.keys() else kwargs["split_kwargs"]

    test_start_date = kwargs["test_start_date"]  # 测试集的第一天
    dtest = data[data.index.get_level_values(0) >= test_start_date]
    # 默认测试集最后一天是数据集的最后一天
    if "test_end_date" in kwargs.keys():
        dtest = dtest[dtest.index.get_level_values(0) <= kwargs["test_end_date"]]
    dtrain = data[~data.index.isin(dtest.index)]

    if split_method == "split_by_date":
        # 默认训练集的第一天是数据集第一天，验证集的第一天是训练集最后一天的第二天
        if "train_start_date" not in split_kwargs.keys():
            train_start_date = dtrain.index.get_level_values(0)[0]
        else:
            train_start_date = split_kwargs["train_start_date"]
        if "train_start_date" not in split_kwargs.keys():
            valid_start_date = datetime.datetime.strptime(split_kwargs["train_end_date"], '%Y-%m-%d')
            valid_start_date += datetime.timedelta(days=1)
            valid_start_date = valid_start_date.strftime('%Y-%m-%d')
        else:
            valid_start_date = split_kwargs["valid_start_date"]
        dtrain, dvalid = split_by_date(dtrain, train_start_date, split_kwargs["train_end_date"], valid_start_date,
                                       split_kwargs["valid_end_date"])
    elif split_method == "split":
        dtrain, dvalid = split(dtrain, split_kwargs)
    else:
        dtrain, dvalid = group_split(dtrain, split_kwargs)
    return dtrain, dvalid, dtest


####################################################
# 自动处理器
####################################################
def process_data(data: pd.DataFrame | pd.Series, norm: str = "z", decay: int = 0,
                 threshold: float = 0.5) -> pd.DataFrame | pd.Series:
    """
    inf_mask -> process_nan -> mad_winsorize -> (decay) -> normalize
    """
    if isinstance(data, pd.Series):
        data = down_sample(data, threshold)
        data = inf_mask(data).dropna()
    else:
        data = ts_ffill(inf_mask(data)).dropna()
    data = mad_winsor(data)
    if decay > 1:
        data = mean(data, ts_decay(data, decay)).dropna()  # 很弱的decay, 不希望给过去太大的权重
    if norm == "z":
        data = cs_zscore(data)
    elif norm == "r":
        data = cs_robust_zscore(data)
    elif norm == "m":
        data = cs_scale(data)
    else:
        data = cs_rank(data)
    return data


def auto_process(X: pd.DataFrame, y: str, norm: str = "z", split_params: dict = None,
                 label_decay: int = 0, unique_threshold: float = 0) -> dict:
    """
    :param X: pd.DataFrame，原始特征，包括了目标值
    :param y: str，目标值所在列的列名
    :param norm: str, 标准化方式, 可选'z'/'r'/'m'
    :param split_params: dict, 划分数据集的方法
    :param label_decay: 是否对目标值做decay以提高unique value的占比
    :param unique_threshold: 降采样的标准
    :return:
    """
    if split_params is None:
        split_params = {
            "data": X,
            "test_start_date": None,
            "split_method": "group_split",
            "split_kwargs": {
                "train": 0.7,
                "valid": 0.3,
            }
        }

    print(X.info())
    X_mis = percentage_missing(X)
    print('X_mis=', X_mis)

    label = X.pop(y)

    print("original label:")
    single_factor_ana(label)

    feature = process_data(X, norm=norm)
    label = process_data(label, norm=norm, decay=label_decay, threshold=unique_threshold)
    print("label processed:")
    single_factor_ana(label)
    print("process dataset done")

    # 拆分数据集
    X_train, X_valid, X_test = split_data_by_date(feature, split_params)
    y_train, y_valid, y_test = split_data_by_date(label, split_params)

    X_train, y_train = align(X_train, y_train)
    X_valid, y_valid = align(X_valid, y_valid)
    X_test, y_test = align(X_test, y_test)

    print("split data done", "\n")

    returns = {
        "X_train": X_train.fillna(0),
        "y_train": y_train,
        "X_valid": X_valid.fillna(0),
        "y_valid": y_valid,
        "X_test": X_test.fillna(0),
        "y_test": y_test,
    }
    print('all works done', '\n')
    return returns


####################################################
# 自动建模（线性回归模型）
####################################################
def auto_lrg(x: pd.DataFrame | pd.Series, y: pd.Series | pd.DataFrame, method: str = "ols", alpha: float = 1e-3,
             max_iter: int = 1000, verbose: int = 1):
    """
    :param x: pd.DataFrame, 特征值
    :param y: pd.Series or pd.DataFrame, 目标值
    :param method: str, 回归方法, 可选'ols', 'lasso', 'ridge'或'logistic'
    :param alpha: 正则化系数
    :param max_iter: int, 最大迭代次数
    :param verbose: int, 等于1时输出使用的线性回归方法
    :return: model
    """
    if isinstance(x, pd.Series):
        x = x.values.reshape(-1, 1)
    model = None
    if verbose == 1:
        print(method + ' method will be used')
    if method == 'ols':
        lrg = linear_model.LinearRegression()
        model = lrg.fit(x, y)
    elif method == 'ridge':
        ridge = linear_model.Ridge(alpha=alpha, max_iter=max_iter)
        model = ridge.fit(x, y)
    elif method == 'lasso':
        lasso = linear_model.Lasso(alpha=alpha, max_iter=max_iter)
        model = lasso.fit(x, y)
    elif method == 'logistic':
        logistic = linear_model.LogisticRegression()
        model = logistic.fit(x, y)
    return model


class LinearRegressionModel:
    """
    线性回归模型封装类，提供与XGBoost类相似的接口
    
    参数:
    task: str, 任务类型，'reg'表示回归，'cls'表示分类（逻辑回归）
    method: str, 回归方法，可选'ols', 'lasso', 'ridge'
    alpha: float, 正则化系数
    max_iter: int, 最大迭代次数
    """
    def __init__(self, lin_model=None, task: str = "reg", method: str = "ols",
                 alpha: float = 1e-3, max_iter: int = 1000):
        self.task = task
        self.method = method
        self.alpha = alpha
        self.max_iter = max_iter
        self.lin_model = lin_model

    def fit(self, x_train: pd.DataFrame, y_train: pd.Series, x_valid: pd.DataFrame = None,
            y_valid: pd.Series = None):
        """
        训练线性模型（保持接口统一性，验证集参数可选）
        """
        from sklearn.linear_model import LinearRegression, Lasso, Ridge
        
        if self.method == 'ols':
            self.lin_model = LinearRegression()
        elif self.method == 'lasso':
            self.lin_model = Lasso(alpha=self.alpha, max_iter=self.max_iter)
        elif self.method == 'ridge':
            self.lin_model = Ridge(alpha=self.alpha, max_iter=self.max_iter)
        
        self.lin_model.fit(x_train, y_train)
        return self

    def predict(self, x_test: pd.DataFrame) -> list:
        """生成预测结果"""
        if self.lin_model is None:
            raise ValueError("模型尚未训练，请先调用fit方法")
        return self.lin_model.predict(x_test).tolist()

    def predict_pandas(self, x: pd.DataFrame) -> pd.Series:
        """生成带索引的预测序列"""
        index = x.index
        # 将列表中的单个元素解包
        pred = pd.Series([item[0] if isinstance(item, list) else item for item in self.predict(x)], 
                        index=index)
        return pred

    def save(self, target_dir: str):
        """保存模型到指定目录"""
        import pickle
        pickle.dump(self.lin_model, open(target_dir + '/linear_model.pkl', 'wb'))

    def load(self, target_dir: str):
        """从目录加载模型"""
        import pickle
        self.lin_model = pickle.load(open(target_dir + '/linear_model.pkl', 'rb'))

    def explain_model(self, index=None):
        """解释模型系数"""
        if self.lin_model is None:
            raise ValueError("模型尚未训练，请先调用fit方法")
            
        print('Linear Model Coefficients:')
        coef = pd.Series(self.lin_model.coef_, index=index)
        print(coef.sort_values(ascending=False))


class hybrid:
    def __init__(self, lin_model=None, xgb_model=None, task: str = "reg", lrg_method: str = "ols", alpha: float = 1e-3,
                 max_iter: int = 1000, xgb_params: dict = None, weight: list = None):
        super(hybrid, self).__init__()
        self.task = task
        self.lrg_method = lrg_method
        self.alpha = alpha
        self.max_iter = max_iter
        self.xgb_params = xgb_params
        self.weight = weight
        self.lin_model = lin_model
        self.xgb_model = xgb_model

    def fit(self, x_train: pd.DataFrame, y_train: pd.Series | pd.DataFrame, x_valid: pd.DataFrame,
            y_valid: pd.Series | pd.DataFrame):
        if self.xgb_params is None:
            est = 800
            eta = 0.0421
            colsamp = 0.9325
            subsamp = 0.8785
            max_depth = 6
            l1 = 0.25
            l2 = 0.5
            early_stopping_rounds = 20
        else:
            est = self.xgb_params['est']
            eta = self.xgb_params['eta']
            colsamp = self.xgb_params['colsamp']
            subsamp = self.xgb_params['subsamp']
            max_depth = self.xgb_params['max_depth']
            l1 = self.xgb_params['l1']
            l2 = self.xgb_params['l2']
            early_stopping_rounds = self.xgb_params['early_stopping_rounds']
        if self.task == 'reg':
            xgb = xgboost.XGBRegressor(objective='reg:squarederror', n_estimators=est, eta=eta,
                                       colsample_bytree=colsamp, subsample=subsamp,
                                       reg_alpha=l1, reg_lambda=l2, max_depth=max_depth,
                                       early_stopping_rounds=early_stopping_rounds)
            self.xgb_model = xgb.fit(x_train, y_train, eval_set=[(x_valid, y_valid)])
        else:
            xgb = xgboost.XGBClassifier(n_estimators=est, eta=eta,
                                        colsample_bytree=colsamp, subsample=subsamp,
                                        reg_alpha=l1, reg_lambda=l2, max_depth=max_depth,
                                        early_stopping_rounds=early_stopping_rounds)
            self.xgb_model = xgb.fit(x_train, y_train, eval_set=[(x_valid, y_valid)])
        self.lin_model = auto_lrg(x_train, y_train, method=self.lrg_method, alpha=self.alpha, max_iter=self.max_iter)

    def predict(self, x_test: pd.DataFrame) -> list:
        if self.weight is None:
            self.weight = [0.4, 0.6]
        pred_x = pd.Series(self.xgb_model.predict(x_test))
        pred_l = pd.Series(self.lin_model.predict(x_test))
        pred = self.weight[0] * pred_l + self.weight[1] * pred_x
        return pred.values

    def predict_pandas(self, x: pd.DataFrame) -> pd.Series:
        index = x.index
        result = []
        result.append(pd.Series(self.predict(x)))
        series = pd.concat(result, axis=0)
        series.index = index
        return series

    def save(self, target_dir: str):
        pickle.dump(self.lin_model, file=open(target_dir + '/linear.pkl', 'wb'))
        pickle.dump(self.xgb_model, file=open(target_dir + '/xgb.pkl', 'wb'))

    def load(self, target_dir: str):
        with open(target_dir + "/linear.pkl", "rb") as file:
            self.lin_model = pickle.load(file)
        file.close()
        with open(target_dir + "/xgb.pkl", "rb") as file:
            self.xgb_model = pickle.load(file)
        file.close()

    def explain_model(self, index):
        print('XGBoost Feature Importance:')
        xgboost.plot_importance(self.xgb_model)
        plt.show()
        importance = self.xgb_model.feature_importances_
        importance = pd.Series(importance, index=index).sort_values(ascending=False)
        print(importance, '\n')
        print('Linear Model Coef:')
        coef = self.lin_model.coef_
        c = pd.Series(coef, index=index).sort_values(ascending=False)
        print(c)

class XGBoost:
    """
    XGBoost模型封装类，提供与hybrid类似的接口
    
    参数:
    task: str, 任务类型，'reg'表示回归，'cls'表示分类
    xgb_params: dict, XGBoost模型参数
    """
    def __init__(self, xgb_model=None, task: str = "reg", xgb_params: dict = None):
        super(XGBoost, self).__init__()
        self.task = task
        self.xgb_params = xgb_params
        self.xgb_model = xgb_model
        
    def fit(self, x_train: pd.DataFrame, y_train: pd.Series | pd.DataFrame, x_valid: pd.DataFrame,
            y_valid: pd.Series | pd.DataFrame):
        """
        训练XGBoost模型
        
        参数:
        x_train: 训练集特征
        y_train: 训练集标签
        x_valid: 验证集特征
        y_valid: 验证集标签
        """
        if self.xgb_params is None:
            # 默认参数
            est = 800
            eta = 0.0421
            colsamp = 0.9325
            subsamp = 0.8785
            max_depth = 6
            l1 = 0.25
            l2 = 0.5
            early_stopping_rounds = 20
        else:
            # 使用用户提供的参数
            est = self.xgb_params.get('est', 800)
            eta = self.xgb_params.get('eta', 0.0421)
            colsamp = self.xgb_params.get('colsamp', 0.9325)
            subsamp = self.xgb_params.get('subsamp', 0.8785)
            max_depth = self.xgb_params.get('max_depth', 6)
            l1 = self.xgb_params.get('l1', 0.25)
            l2 = self.xgb_params.get('l2', 0.5)
            early_stopping_rounds = self.xgb_params.get('early_stopping_rounds', 20)
            
        if self.task == 'reg':
            xgb = xgboost.XGBRegressor(
                objective='reg:squarederror', 
                n_estimators=est, 
                eta=eta,
                colsample_bytree=colsamp, 
                subsample=subsamp,
                reg_alpha=l1, 
                reg_lambda=l2, 
                max_depth=max_depth,
                early_stopping_rounds=early_stopping_rounds
            )
            self.xgb_model = xgb.fit(x_train, y_train, eval_set=[(x_valid, y_valid)])
        else:
            xgb = xgboost.XGBClassifier(
                n_estimators=est, 
                eta=eta,
                colsample_bytree=colsamp, 
                subsample=subsamp,
                reg_alpha=l1, 
                reg_lambda=l2, 
                max_depth=max_depth,
                early_stopping_rounds=early_stopping_rounds
            )
            self.xgb_model = xgb.fit(x_train, y_train, eval_set=[(x_valid, y_valid)])
            
    def predict(self, x_test: pd.DataFrame) -> list:
        """
        使用训练好的模型进行预测
        
        参数:
        x_test: 测试集特征
        
        返回:
        预测结果列表
        """
        if self.xgb_model is None:
            raise ValueError("模型尚未训练，请先调用fit方法")
        
        pred = pd.Series(self.xgb_model.predict(x_test))
        return pred.values
    
    def predict_pandas(self, x: pd.DataFrame) -> pd.Series:
        """
        使用训练好的模型进行预测，并返回pandas.Series
        
        参数:
        x: 测试集特征
        
        返回:
        预测结果Series，保留原索引
        """
        index = x.index
        result = []
        result.append(pd.Series(self.predict(x)))
        series = pd.concat(result, axis=0)
        series.index = index
        return series
    
    def save(self, target_dir: str):
        """
        保存模型到指定目录
        
        参数:
        target_dir: 目标目录
        """
        pickle.dump(self.xgb_model, file=open(target_dir + '/xgb_model.pkl', 'wb'))
    
    def load(self, target_dir: str):
        """
        从指定目录加载模型
        
        参数:
        target_dir: 目标目录
        """
        with open(target_dir + "/xgb_model.pkl", "rb") as file:
            self.xgb_model = pickle.load(file)
        file.close()
    
    def explain_model(self, index=None):
        """
        解释模型，展示特征重要性
        
        参数:
        index: 特征名称列表，默认为None
        """
        if self.xgb_model is None:
            raise ValueError("模型尚未训练，请先调用fit方法")
            
        print('XGBoost Feature Importance:')
        xgboost.plot_importance(self.xgb_model)
        plt.show()
        
        importance = self.xgb_model.feature_importances_
        if index is not None:
            importance = pd.Series(importance, index=index).sort_values(ascending=False)
            print(importance)


def auto_lgbm(x_train: pd.DataFrame, y_train: pd.Series | pd.DataFrame, x_valid: pd.DataFrame,
              y_valid: pd.Series | pd.DataFrame, early_stopping: int = 30, verbose_eval: int = 20,
              lgb_params: dict = None, num_boost_round: int = 1000, evals_result: dict = None,
              explain=False):
    if evals_result is None:
        evals_result = {}
    if lgb_params is None:
        lgb_params = {
            "loss": "mse",
            "colsample_bytree": 0.8879,
            "learning_rate": 0.0421,
            "subsample": 0.8789,
            "lambda_l1": 205.6999,
            "lambda_l2": 580.9768,
            "max_depth": 8,
            "num_leaves": 210,
            "num_threads": 20,
            "verbosity": -1
        }
    dtrain = lgb.Dataset(x_train, label=y_train)
    dvalid = lgb.Dataset(x_valid, label=y_valid)
    early_stopping_callback = lgb.early_stopping(early_stopping)
    verbose_eval_callback = lgb.log_evaluation(period=verbose_eval)
    evals_result_callback = lgb.record_evaluation(evals_result)
    model = lgb.train(
        params=lgb_params,
        train_set=dtrain,
        num_boost_round=num_boost_round,
        valid_sets=[dtrain, dvalid],
        valid_names=["train", "valid"],
        callbacks=[early_stopping_callback, verbose_eval_callback, evals_result_callback],
    )
    if explain:
        lgb.plot_importance(model)
    return model


####################################################
# 评估指标
####################################################
def cov(x: np.array, y: np.array) -> float:
    x_bar = x.mean()
    y_bar = y.mean()
    cov_xy = 0
    for i in range(0, len(x)):
        cov_xy += (x[i] - x_bar) * (y[i] - y_bar)
    cov_xy = cov_xy / len(x)
    return cov_xy


def pearson_corr(x, y) -> float:
    np.array(x)
    np.array(y)
    x_std = x.std()
    y_std = y.std()
    cov_xy = cov(x, y)
    cor = cov_xy / (x_std * y_std)
    return cor


def ic_ana(pred: pd.Series | pd.DataFrame, y: pd.DataFrame | pd.Series, groupby: str = None, plot: bool = True,
           freq: int = 30) -> tuple[float, float, float, float, float, float]:
    """
    :param pred: pd.DataFrame or pd.Series, 预测值
    :param y: pd.DataFrame or pd.Series, 真实值
    :param groupby: str, 排序依据
    :param plot: bool, 控制是否画出IC曲线
    :param freq: int, 频率, 用于平滑IC序列
    :return: float, 依次为ic均值, icir, rank_ic均值和rank_icir
    """
    groupby = pred.index.names[0] if groupby is None else groupby
    concat_data = pd.concat([pred, y], axis=1)
    ic = concat_data.groupby(groupby).apply(lambda x: x.iloc[:, 0].corr(x.iloc[:, 1]))
    rank_ic = concat_data.groupby(groupby).apply(lambda x: x.iloc[:, 0].corr(x.iloc[:, 1], method='spearman'))
    if plot:
        ic.index = pd.to_datetime(ic.index)
        rank_ic.index = pd.to_datetime(rank_ic.index)
        # 默认freq为30的情况下，画出来的IC是月均IC
        plt.figure(figsize=(10, 6))
        plt.plot(ic.rolling(freq).mean(), label='ic', marker='o')
        plt.plot(rank_ic.rolling(freq).mean(), label='rank_ic', marker='o')
        plt.ylabel('score')
        plt.title('IC Series (rolling ' + str(freq) + ')')
        plt.legend()
        plt.show()
        plt.clf()
        q.show_dist(ic)
    print(ic.describe())
    IC, ICIR, Rank_IC, Rank_ICIR = ic.mean(), ic.mean() / ic.std(), rank_ic.mean(), rank_ic.mean() / rank_ic.std()
    t_stat = ICIR * (len(y) ** 0.5)
    win_rate = (ic>0).mean()
    return IC, ICIR, Rank_IC, Rank_ICIR, t_stat, win_rate

def ts_ic_ana(pred: pd.Series | pd.DataFrame, y: pd.DataFrame | pd.Series, plot: bool = True,
           freq: int = 30) -> tuple[float, float]:
    """
    用于时间序列数据的ic分析
    :param pred: pd.DataFrame or pd.Series, 预测值
    :param y: pd.DataFrame or pd.Series, 真实值
    :param groupby: str, 排序依据
    :param plot: bool, 控制是否画出IC曲线
    :param freq: int, 频率, 用于平滑IC序列
    :return: float, 依次为ic均值, icir, rank_ic均值和rank_icir
    """
    concat_data = pd.concat([pred, y], axis=1)
    # 计算 IC
    ic = concat_data.iloc[:, 0].corr(concat_data.iloc[:, 1])

    ic_series = concat_data.iloc[:, 0].rolling(window=freq).corr(concat_data.iloc[:, 1])
    print(ic_series.tail())
    
    # 计算 ICIR (Information Coefficient Information Ratio)
    icir = ic_series.mean() / ic_series.std() if ic_series.std() != 0 else 0
    
    if plot:
        # ic.index = pd.to_datetime(ic.index)
        # rank_ic.index = pd.to_datetime(rank_ic.index)
        # 默认freq为30的情况下，画出来的IC是月均IC
        plt.figure(figsize=(10, 6))
        plt.plot(ic_series, label='ic', marker='o')
        plt.ylabel('score')
        plt.title('IC Series (rolling ' + str(freq) + ')')
        plt.legend()
        plt.show()
        plt.clf()
        show_dist(ic_series)
    IC, ICIR = ic, icir
    return IC, ICIR
