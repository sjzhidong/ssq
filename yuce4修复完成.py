import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.metrics import mean_squared_error

# 导入数据并指定数据类型
data = pd.read_csv('data.csv', dtype={'red1': int, 'red2': int, 'red3': int, 'red4': int, 'red5': int, 'red6': int, 'blue': int})

# 定义数据预处理函数
def preprocess_data(data):
    preprocessed_data = (data - 1) / np.array([33, 33, 33, 33, 33, 33, 16])
    return preprocessed_data

# 数据预处理
data[['red1', 'red2', 'red3', 'red4', 'red5', 'red6', 'blue']] = preprocess_data(data[['red1', 'red2', 'red3', 'red4', 'red5', 'red6', 'blue']])

# 将数据划分为训练集和测试集
train_data = data.iloc[:1500, :]
test_data = data.iloc[1500:, :]

# 定义函数来生成训练和测试数据
def generate_data(data, lookback):
    X, Y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:i+lookback, :])
        Y.append(data[i+lookback, :])
    return np.array(X), np.array(Y)

# 设置LSTM模型参数
lookback = 10
batch_size = 32
epochs = 200

# 生成训练和测试数据
train_X, train_Y = generate_data(train_data[['red1', 'red2', 'red3', 'red4', 'red5', 'red6', 'blue']].to_numpy(), lookback)
test_X, test_Y = generate_data(test_data[['red1', 'red2', 'red3', 'red4', 'red5', 'red6', 'blue']].to_numpy(), lookback)

# 创建LSTM模型
model = Sequential()
model.add(LSTM(64, input_shape=(lookback, 7)))
model.add(Dense(32, activation='relu'))
model.add(Dense(7, activation='sigmoid'))
model.compile(loss='mean_squared_error', optimizer='adam')

# 训练模型
model.fit(train_X, train_Y, batch_size=batch_size, epochs=epochs, validation_data=(test_X, test_Y))

# 预测下一期双色球
last_data = data[['red1', 'red2', 'red3', 'red4', 'red5', 'red6', 'blue']].tail(lookback).to_numpy()
last_data_reshaped = np.reshape(last_data, (1, lookback, last_data.shape[1]))

predictions = []
probabilities = []
for _ in range(5):
    prediction = model.predict(last_data_reshaped)
    prediction = np.round(prediction * np.array([33, 33, 33, 33, 33, 33, 16])) + np.array([1, 1, 1, 1, 1, 1, 1])

    # 处理重复的红球号码
    red_balls = np.unique(prediction[0][:6]) # 使用np.unique去除重复的号码，并按升序排列
    while len(red_balls) < 6: # 如果去重后的号码不足6个，需要补充随机的号码，直到满足6个为止
        remaining_numbers = np.setdiff1d(np.arange(1, 34), red_balls) # 获取剩余的可选号码
        replacement = np.random.choice(remaining_numbers) # 随机选择一个补充的号码
        red_balls = np.append(red_balls,replacement) # 将补充的号码加入到红球结果中

    predictions.append(prediction)

    prob = model.predict(last_data_reshaped)
    prob = np.round(prob * 100, 2)
    probabilities.append(prob)

    last_data = np.concatenate((last_data[1:], prediction), axis=0)
    last_data_reshaped = np.reshape(last_data, (1, lookback, last_data.shape[1]))

print("下一期双色球可能的预测结果为：")
for i in range(len(predictions)):
    print("预测结果", i + 1, ":")

    # 获取模型预测的红球结果
    red_balls = predictions[i][0][:6]

    # 获取模型预测的蓝球结果
    blue_ball = predictions[i][0][6]
    blue_ball = int(blue_ball) # 对蓝球结果进行取整

    print("红球号码：", red_balls)
    print("蓝球号码：", blue_ball)
    print("可能性：")
    for j in range(len(probabilities[i][0])):
        print("  第%d位号码概率：%f" % (j+1, probabilities[i][0][j]))