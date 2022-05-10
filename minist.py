# -*- coding: utf-8 -*-

# In[1]:
# import
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import torchvision.transforms as transforms
from tqdm import tqdm

from predata import *
from env import *
from utils import *
from net import CNN


# In[3]
# 导入数据
convert_data()            # 将数据集转换成csv
data_train = data_train() # 训练集导入
data_test = data_test()   # 数据集导入
dataloader_train = DataLoader(dataset=data_train, batch_size=64, shuffle=True) # 训练集装载
dataloader_test = DataLoader(dataset=data_test, batch_size=64, shuffle=True)   # 数据集装载


# In[]
# 数据预览
images, labels = next(iter(dataloader_train))
img = make_grid(images)
img = img.numpy().transpose(1, 2, 0)
mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]
img = img * std + mean
print([labels[i] for i in range(16)])
plt.imshow(img)
plt.show()

# In[5]
# 训练和参数优化
cnn = CNN()
if torch.cuda.is_available(): # 判断是否有可用的 GPU 以加速训练
    cnn = cnn.cuda()
loss_F = nn.CrossEntropyLoss() # 设置损失函数为 CrossEntropyLoss（交叉熵损失函数）
optimizer = torch.optim.Adam(cnn.parameters(), lr=lr) # 设置优化器为 Adam 优化器

# 训练
print('train...', end='\n')
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
        print(outputs.data)
        print(y_train.data)
        break
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
    print("Loss: {:.4f}  Train Accuracy: {:.4f}%  Test Accuracy: {:.4f}%".format(
        running_loss / len(data_train),
        100 * running_correct / len(data_train),
        100 * testing_correct / len(data_test))
    )

torch.save(cnn, './data/model.pth')  # 保存模型