# -*- coding: utf-8 -*-

import csv
import os
from tqdm import tqdm
import torch

# 将二进制数据文件img_file和标签文件label_file转换为csv文件out_file
def convert(img_file, label_file, out_file, n):
    if os.path.isfile(out_file):
        return

    f = open(img_file, "rb")
    l = open(label_file, "rb")
    o = open(out_file, "w")

    f.read(16)
    l.read(8)
    images = []

    print("Generator " + out_file + " ...")
    for i in tqdm(range(n)):
        image = [ord(l.read(1))]
        for j in range(28 * 28):
            image.append(ord(f.read(1)))
        images.append(image)

    for image in images:
        o.write(",".join(str(pix) for pix in image) + "\n")

    f.close()
    o.close()
    l.close()

# 采样一个数据
def sample_data(source):
    csv_file = csv.reader(open(source))

    for content in csv_file:
        content = list(map(float, content))
        feature = content[1:785]
        return feature

# 加载所有的数据并作归一化
def load_data(source):
    csv_file = csv.reader(open(source))
    data = []
    for content in csv_file:
        content = list(map(float, content))
        feature = torch.tensor(content[1:785]).reshape(1,28,28)
        feature = feature / 255 * 2 - 1
        label = int(content[0])
        data.append((feature, label))
    return data

# 转换成pytorch变量
def get_Variable(x):
    x = torch.autograd.Variable(x)
    return x.cuda() if torch.cuda.is_available() else x