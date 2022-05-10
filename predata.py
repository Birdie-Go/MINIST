# -*- coding: utf-8 -*-

from torch.utils.data import Dataset
import numpy as np
from PIL import Image

from utils import *
from env import *

# 训练集
class data_train(Dataset):
    def __init__(self):
        self.x = load_data(data_path)

    def __getitem__(self, idx):
        assert idx < len(self.x)
        return self.x[idx]

    def __len__(self):
        return len(self.x)

# 测试集
class data_test(Dataset):
    def __init__(self):
        self.x = load_data(test_path)

    def __getitem__(self, idx):
        assert idx < len(self.x)
        return self.x[idx]

    def __len__(self):
        return len(self.x)

# 将数据从gz转换成csv，并采样一个进行展示
def convert_data():  
    convert(train_images_path, train_labels_path, data_path, 60000)
    convert(test_images_path, test_labels_path, test_path, 10000)
    print("Convert Finished!")

    feature = sample_data(data_path)
    # print('feature:', feature)

    b = np.array(feature).reshape(28,28)
    img = Image.fromarray(np.uint8(b))
    img.show() 
