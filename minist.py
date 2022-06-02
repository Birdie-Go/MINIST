# -*- coding: utf-8 -*-

# import
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

from predata import *
from env import *
from utils import *
from net import CNN

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument("--method", type=str, default="GD")
parser.add_argument("--batch_size", type=int, default=64)
args = parser.parse_args()
method = args.method
batch_size = args.batch_size


# 导入数据
convert_data()            # 将数据集转换成csv
data_train = data_train() # 训练集导入
data_test = data_test()   # 数据集导入
print("Loading...")
if method == 'GD':
    dataloader_train = DataLoader(dataset=data_train) # 训练集装载
    dataloader_test = DataLoader(dataset=data_test)   # 数据集装载
elif method == 'SGD' or method == 'SAG':
    dataloader_train = DataLoader(dataset=data_train, batch_size=batch_size, shuffle=True) # 训练集装载
    dataloader_test = DataLoader(dataset=data_test, batch_size=batch_size, shuffle=True)   # 数据集装载

# 训练和参数优化
cnn = CNN()
loss_F = nn.CrossEntropyLoss() # 设置损失函数为 CrossEntropyLoss（交叉熵损失函数）
if method == 'GD':
    optimizer = torch.optim.Adam(cnn.parameters(), lr=lr) # 设置优化器为 Adam 优化器
elif method == 'SGD':
    optimizer = torch.optim.SGD(cnn.parameters(), lr=lr) # 设置优化器为 SGD 优化器
else:
    optimizer = torch.optim.ASGD(cnn.parameters(), lr=lr) # 设置优化器为 ASGD 优化器
Draw = Drawer(method)

# 训练
print('train...', end='\n')
running_losses = []
testing_corrects = []
running_corrects = []
messages = []
for epoch in range(epochs):
    # 训练
    running_loss = 0.0     # 一个 epoch 的损失
    running_correct = 0.0  # 一个 epoch 中所有训练数据的准确率
    print("Epoch [{}/{}]".format(epoch, epochs)) # 打印当前的进度 当前epoch/总epoch数
    for data in tqdm(dataloader_train): # 遍历每个数据，并使用tqdm反应训练进度
        X_train, y_train = data # data是一个tuple，第一维是训练数据，第二维标签
        X_train, y_train = get_Variable(X_train), get_Variable(y_train) # 将数据变成pytorch需要的变量
        outputs = cnn(X_train) # 将数据输入进入网络，得到输出结果
        _, pred = torch.max(outputs.data, 1) # 输出的结果是一个大小为10的数组
                                             # 我们获取最大值和最大值的索引，后者表示预测结果
        optimizer.zero_grad() # 梯度置零
        loss = loss_F(outputs, y_train) # 计算输出结果和标签损失
        loss.backward() # 根据梯度反向传播
        optimizer.step() # 根据梯度更新所有的参数
        running_loss += loss.item()  # 累计全局的损失
        running_correct += torch.sum(pred == y_train.data) # 计算准确率
    
    # 测试
    testing_correct = 0.0
    for data in dataloader_test:
        X_test, y_test = data
        X_test, y_test = get_Variable(X_test), get_Variable(y_test)
        outputs = cnn(X_test)
        _, pred = torch.max(outputs, 1)
        testing_correct += torch.sum(pred == y_test.data)
    
    # 打印信息
    message = "Loss: {:.4f}  Train Accuracy: {:.4f}%  Test Accuracy: {:.4f}%".format(
        running_loss / len(data_train),
        100 * running_correct / len(data_train),
        100 * testing_correct / len(data_test)
    )
    print(message)
    messages.append(message)

    # 画图
    running_losses.append(running_loss / len(data_train))
    running_corrects.append(100 * running_correct / len(data_train))
    testing_corrects.append(100 * testing_correct / len(data_train))
    Draw.draw(running_losses, title="train_loss")
    Draw.draw(running_corrects, ylabel="%", title="train_accuracy")
    Draw.draw(testing_corrects, ylabel="%", title="test_accuracy")

torch.save(cnn, './data/model_{}.pth'.format(method))  # 保存模型
with open("output_{}.txt".format(method), "w") as f:
    for message in messages:
        f.write(message + '\n')
f.close()