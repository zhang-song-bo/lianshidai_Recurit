import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def pre():
    """训练前的数据处理"""
    # 加载数据
    df = pd.read_csv("D:/lianshidaiRecurit/03/PSCompPars.csv")

    # 处理缺失值
    df = df.dropna(subset=['pl_orbsmax', 'st_mass', 'pl_orbper'])

    # 提取相关特征和目标变量
    X = df[['pl_orbsmax', 'st_mass']]
    y = df['pl_orbper']

    # 进行 log10 对数变换
    X_log10 = np.log10(X)
    y_log10 = np.log10(y)

    # 拆分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X_log10, y_log10, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

def sklearn(X_train, X_test, y_train, y_test):
    # 使用 sklearn 进行线性回归
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 预测
    y_pred_log = model.predict(X_test)

    # 反变换回原始尺度
    y_pred = 10 ** y_pred_log

    # 计算 sklearn 线性回归的 R^2 和 MSE,注意在 log10 尺度计算 R^2 ，MSE 需要在原尺度计算
    r2_sklearn = r2_score(y_test, y_pred_log)
    mse_sklearn = mean_squared_error(10 ** y_test, y_pred)

    return r2_sklearn, mse_sklearn, model, y_pred

def output():
    # 输出结果
    print("Sklearn 线性回归模型：")
    print(f"权重: {intercept_sklearn}")
    print(f"偏置: {coef_sklearn}")
    print(f"R²: {r2_sklearn}")
    print("log10(MSE):", log_mse_sklearn)
    print(f"Sklearn: log10(y) = {model.intercept_:.4f} + {model.coef_[0]:.4f} * log10(pl_orbsmax) + {model.coef_[1]:.4f} * log10(st_mass)")

def draw_plot_map():
    # 可视化预测结果
    plt.scatter(10 ** y_test, y_pred, label="Sklearn Predict", alpha=0.5)
    plt.plot([(10 ** y_test).min(), (10 ** y_test).max()], [(10 ** y_test).min(), (10 ** y_test).max()], 'r--')
    plt.xlabel("True orbital period")
    plt.ylabel("Predicted orbital period")
    plt.title("Prediction vs True value (Log Transformed)")
    plt.legend()
    plt.show()


if __name__ == "__main__" :
    X_train, X_test, y_train, y_test = pre()
    r2_sklearn, mse_sklearn, model, y_pred = sklearn(X_train, X_test, y_train, y_test)

    log_mse_sklearn = np.log10(mse_sklearn)

    # 获取回归系数
    intercept_sklearn = model.intercept_
    coef_sklearn = model.coef_

    output()
    draw_plot_map()
