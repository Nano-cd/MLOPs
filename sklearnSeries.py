import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn

# Enable MLflow's automatic experiment tracking for scikit-learn
mlflow.sklearn.autolog()
mlflow.autolog()
mlflow.set_experiment("fit_curve_find")


def f_fit(x, a, b, c, d):
    return (a - d) / (1 + (x / c) ** b) + d


# A, B, D是特征，C是目标值
# A数据
A = np.array([40, 50, 60, 70])

# B数据
B = np.array([105, 115, 125])

# # D数据
# D = np.array([0, 1, 2, 3, 4, 5])

# C数据
# # 将数据用,隔开
# result = ','.join(map(str, C.flatten()))
# print(result)
# # End Generation Here

#C_0
C_0 = np.array([
    843196.5, 898756, 1040538.5, 1194694.5,
    809270.5, 966908.5, 1092221, 1268875,
    890684, 989262, 1128462.5, 1330357
])
# C_1
C_1 = np.array([
    238503.5, 257733, 309053.5, 376744.5,
    252355.5, 264417, 322108.5, 379868,
    279376.5, 301531, 347916.5, 439815.5
])

# C_2
C_2 = np.array([
    111223.5, 127621.5, 149721, 187953.5,
    117026, 130885, 162363.5, 193150.5,
    130757.5, 147616.5, 171794.5, 208726
])

# C_3
C_3 = np.array([
    59056.5, 63302, 76769.5, 92289,
    57482, 68971.5, 79754, 98814,
    62318.5, 72737, 84576.5, 103800
])

# C_4
C_4 = np.array([
    33439.5, 36224.5, 42689, 53115,
    31412, 37150.5, 45598, 55886,
    33895, 37918, 46382, 56282.5
])

# C_5
C_5 = np.array([
    21756, 24228.5, 28548.5, 36147,
    20177.5, 24808.5, 30129.5, 37928.5,
    20859.5, 25539.5, 29484, 37531.5
])

C = C_5

# 将A, B, D合并为特征矩阵X
# 生成X
X = np.array([[a, b] for a in A for b in B])

# 分割数据集为训练集和测试集
X_train, X_test, C_train, C_test = train_test_split(X, C, test_size=0.1, random_state=42)

# 创建线性回归模型
model = LinearRegression()
# 在模型训练前设置mlflow实验
mlflow.set_experiment("Linear Regression Model")

with mlflow.start_run():
    # 记录模型参数
    mlflow.log_param("test_size", 0.1)
    mlflow.log_param("random_state", 42)

    # 训练模型（原有代码保持不变）
    # 训练模型
    model.fit(X_train, C_train)



    # 记录线性回归模型
    mlflow.sklearn.log_model(model, "linear_regression_model")
    # 预测
    C_pred = model.predict(X_test)

    # 评估模型
    mse = mean_squared_error(C_test, C_pred)
    print(f"Mean Squared Error: {mse}")
    # 将数据点画出来，并区分出测试label和实际预测点
    plt.scatter(X_train[:, 0], C_train, color='blue', label='train_dataset')
    plt.scatter(X_test[:, 0], C_test, color='green', label='test_label')
    plt.scatter(X_test[:, 0], C_pred, color='red', label='test_pridicted', marker='x')
    plt.xlabel('Feature_A')
    plt.ylabel('Target_C')
    plt.legend()
    plt.title('fit results')
    plt.show()

    # 保存可视化图表
    plt.savefig("regression_plot.png")
    mlflow.log_artifact("regression_plot.png")
    # 记录模型指标
    mlflow.log_metric("mse", mse)




# # 使用最小二乘法求解
# from scipy.optimize import leastsq
# import numpy as np
#
#
# # 定义误差函数
# def f_err_linear(params, X, C):
#     m, n, c = params
#     predicted_C = m * X[:, 0] + n * X[:, 1] + c
#     return C - predicted_C
#
#
# # 使用最小二乘法进行拟合
# outcome, _ = leastsq(f_err_linear, [1, 1, 1], args=(X, C), maxfev=900000)
#
# a, b, d = outcome
# print(f"拟合参数: a = {a}, b = {b}, d = {d}")
#
# # 计算拟合的C值
# C_fit = a * X[:, 0] + b * X[:, 1] + d
# # 可视化数据点和拟合结果
# plt.scatter(X[:, 0], C, color='blue', label='ori')
# plt.scatter(X[:, 0], C_fit, color='green', label='pred')
# plt.xlabel('Feature_A')
# plt.ylabel('target_C')
# plt.legend()
# plt.title('fit results')
# plt.show()
