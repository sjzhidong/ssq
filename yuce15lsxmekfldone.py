import numpy as np
import pandas as pd

# 导入数据并指定数据类型
data = pd.read_csv('data.csv', dtype={'red1': int, 'red2': int, 'red3': int, 'red4': int, 'red5': int, 'red6': int, 'blue': int})

# 定义数据预处理函数
def preprocess_data(data):
    preprocessed_data = data / np.array([33, 33, 33, 33, 33, 33, 16])
    return preprocessed_data

# 数据预处理
data[['red1', 'red2', 'red3', 'red4', 'red5', 'red6', 'blue']] = preprocess_data(data[['red1', 'red2', 'red3', 'red4', 'red5', 'red6', 'blue']])

# 将数据划分为训练集和测试集
train_data = data.iloc[:1500, :]
test_data = data.iloc[1500:, :]

# 定义函数来生成训练和测试数据
def generate_data(data):
    X = data[:-1]
    Y = data[1:]
    return np.array(X), np.array(Y)

# 生成训练和测试数据
train_X, train_Y = generate_data(train_data[['red1', 'red2', 'red3', 'red4', 'red5', 'red6', 'blue']].to_numpy())
test_X, test_Y = generate_data(test_data[['red1', 'red2', 'red3', 'red4', 'red5', 'red6', 'blue']].to_numpy())

# 定义函数来计算转移概率矩阵，即每个号码从上一期到下一期的转移概率
def get_transition_matrix(X, Y):
    matrix = np.eye(34) # 初始化转移概率矩阵，每一行对应一个位置，每一列对应一个号码，元素为转移概率
    for i in range(len(X)):
        for j in range(7): # 遍历每个位置
            x = X[i][j] # 获取上一期的号码
            y = Y[i][j] # 获取下一期的号码
            matrix[j][int(y * 33)] += 1 # 在对应的位置和号码上累加出现次数
    matrix = matrix / len(X) # 将出现次数除以总样本数，得到转移概率
    return matrix

# 计算训练集的转移概率矩阵
transition_matrix = get_transition_matrix(train_X, train_Y)

# 打印转移概率矩阵
print("转移概率矩阵为：")
print(transition_matrix)

# 预测下一期双色球
last_data = data[['red1', 'red2', 'red3', 'red4', 'red5', 'red6', 'blue']].tail(1).to_numpy()
predictions = [] # 定义一个空的列表，用来存储每一期的预测结果
for _ in range(1):
    prediction = []
    for i in range(7): # 遍历每个位置
        x = last_data[0][i] # 获取上一期的号码
        prob = transition_matrix[i] # 获取该位置的转移概率向量
        prob = prob / prob.sum()
        y = np.random.choice(np.arange(34), size=7, replace=False, p=prob) # 根据转移概率向量，一次性选择七个号码作为预测结果
        prediction.append(y / 33) # 将预测结果还原为归一化后的范围

    predictions.append(prediction)

    # 将最近的一期数据更新为包含预测结果的数据
    last_data = np.array([prediction])

print("下一期双色球可能的预测结果为：")
for i in range(len(predictions)):
    print("预测结果", i + 1, ":")
    # 将预测结果还原为原始的号码范围
    prediction = np.array(prediction)
    prediction = prediction.flatten()
    prediction = np.round(np.array(predictions[i]) * np.array([33, 33, 33, 33, 33, 33, 16])) + np.array([1, 1, 1, 1, 1, 1, 1])
    prediction = prediction.flatten()
    print("红球号码：", prediction[:6])
    # 对蓝球结果进行取整
    blue_ball = int(prediction[6])
    print("蓝球号码：", blue_ball)
