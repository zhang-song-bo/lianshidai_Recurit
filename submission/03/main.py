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

    # 拆分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

def sklearn(X_train, X_test, y_train, y_test):
    # 使用 sklearn 进行线性回归
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 预测
    y_pred = model.predict(X_test)

    # 计算 sklearn 线性回归的 R^2 和 MSE
    r2_sklearn = r2_score(y_test, y_pred)
    mse_sklearn = mean_squared_error(y_test, y_pred)

    return r2_sklearn, mse_sklearn, model, y_pred

def manual(X_train, X_test, y_train, y_test):
    """手动实现最小二乘法计算参数"""
    # 偏置项
    X_b = np.c_[np.ones((len(X_train), 1)), X_train.values]
    # 伪逆法求解 theta_best
    theta_best = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y_train

    # 手动预测函数
    def predict_manual(X):
        X = X.values  # 转换为 NumPy 数组
        X_b_test = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b_test @ theta_best

    # 计算手动最小二乘法的预测值
    y_manual_pred = predict_manual(X_test)

    # 计算手动实现的 R^2 和 MSE
    r2_manual = r2_score(y_test, y_manual_pred)
    mse_manual = mean_squared_error(y_test, y_manual_pred)

    return r2_manual, mse_manual, theta_best, y_manual_pred

def output():
    # 输出结果
    print("Sklearn 线性回归模型：")
    print(f"权重: {intercept_sklearn}")
    print(f"偏置: {coef_sklearn}")
    print(f"R²: {r2_sklearn}")
    print("log10(MSE):", log_mse_sklearn)
    print(f"Sklearn: y = {model.intercept_:.4f} + {model.coef_[0]:.4f} * pl_orbsmax + {model.coef_[1]:.4f} * st_mass")
    print("------------------------------------------------------")
    print("手动最小二乘法模型：")
    print(f"计算出的 theta_best: {theta_best.flatten()}")
    print(f"R²: {r2_manual}")
    print("log10(MSE):", log_mse_manual)
    print(f"Manual: y = {theta_best[0]:.4f} + {theta_best[1]:.4f} * pl_orbsmax + {theta_best[2]:.4f} * st_mass")

def draw_plot_map():
    # 可视化预测结果
    plt.scatter(y_test, y_pred, label="Sklearn Predict", alpha=0.5)
    plt.scatter(y_test, y_manual_pred, label="Manual Least Squares Predict", alpha=0.5, marker="x")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel("True orbital period")
    plt.ylabel("Predicted orbital period")
    plt.title("Prediction vs True value (Log Transformed)")
    plt.legend()
    plt.show()

if __name__=="__main__":
    X_train, X_test, y_train, y_test = pre()
    r2_sklearn, mse_sklearn, model, y_pred = sklearn(X_train, X_test, y_train, y_test)
    r2_manual, mse_manual, theta_best, y_manual_pred = manual(X_train, X_test, y_train, y_test)

    # 计算 MSE 时使用 log10 ，避免数值过大
    log_mse_sklearn = np.log10(mse_sklearn)
    log_mse_manual = np.log10(mse_manual)

    # 获取回归系数
    intercept_sklearn = model.intercept_
    coef_sklearn = model.coef_
    intercept_manual = theta_best[0]
    coef_manual = theta_best[1:]

    output()
    draw_plot_map()