import sys
import cv2 as cv

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import QFileDialog, QMainWindow

from demo01 import Ui_MainWindow

import os
import torch
import torchvision
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from PIL import Image


# 卷积网络
class EffNet(nn.Module):

    def __init__(self, nb_classes=6, include_top=True, weights=None):
        super(EffNet, self).__init__()

        self.block1 = self.make_layers(32, 64)
        self.block2 = self.make_layers(64, 128)
        self.block3 = self.make_layers(128, 256)
        self.linear = nn.Linear(in_features=256 * 25 * 25, out_features=6, bias=False)  # batch_size
        self.include_top = include_top
        self.weights = weights

    def make_layers(self, ch_in, ch_out):
        layers = [
            nn.Conv2d(3, ch_in, kernel_size=(1, 1), stride=(1, 1), bias=False, padding=0,
                      dilation=(1, 1)) if ch_in == 32 else nn.Conv2d(ch_in, ch_in, kernel_size=(1, 1), stride=(1, 1),
                                                                     bias=False, padding=0, dilation=(1, 1)),
            self.make_post(ch_in),

            # 2维深度卷积，用的一个1x3的空间可分离卷积
            nn.Conv2d(ch_in, 1 * ch_in, groups=ch_in, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1), bias=False,
                      dilation=(1, 1)),
            self.make_post(ch_in),

            # 最大池化
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),

            # 2维深度卷积，用的一个3x1的空间可分离卷积
            nn.Conv2d(ch_in, 1 * ch_in, groups=ch_in, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0), bias=False,
                      dilation=(1, 1)),
            self.make_post(ch_in),

            nn.Conv2d(ch_in, ch_out, kernel_size=(1, 2), stride=(1, 2), bias=False, padding=(0, 0), dilation=(1, 1)),
            self.make_post(ch_out),
        ]
        return nn.Sequential(*layers)

    def make_post(self, ch_in):
        layers = [
            nn.LeakyReLU(0.3),
            nn.BatchNorm2d(ch_in, momentum=0.99)
        ]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.block1(x)
        # x = nn.functional.dropout(x, p=0.5, training=self.training)
        x = self.block2(x)
        x = self.block3(x)
        if self.include_top:
            x = x.view(-1, 256 * 25 * 25)
            # print(x.shape)
            x = self.linear(x)
        return x

#运行设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.get_device_name(0))

#数据集正则化
normalize = transforms.Normalize(mean=[0.4, 0.4, 0.4], std=[0.2, 0.2, 0.2])
transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(200),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3,1,1)), #添加这行
            normalize
            ])


net = EffNet().cuda()
net.load_state_dict(torch.load('MyProject.pt'))
net = net.to(device)
torch.no_grad()
classes = ['C1_inclusion','C2_patches','C3_crazing','C4_pitted','C5_rolled-in','C6_scratches']

#GUI界面设置
class PyQtMainEntry(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        
    def btnReadImage_Clicked(self):
        '''
        从本地读取图片
        '''
        # 打开文件选取对话框
        global filename
        filename,  _ = QFileDialog.getOpenFileName(self, '打开图片')
        self.captured = cv.imread(filename)
        # OpenCV图像以BGR通道存储，显示时需要从BGR转到RGB
        self.captured = cv.cvtColor(self.captured, cv.COLOR_BGR2RGB)

        rows, cols, channels = self.captured.shape
        bytesPerLine = channels * cols
        QImg = QImage(self.captured.data, cols, rows, bytesPerLine, QImage.Format_RGB888)
        self.labelCapture.setPixmap(QPixmap.fromImage(QImg).scaled(
            self.labelCapture.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            
    def btnIdentify_Clicked(self):
        '''
        识别
        '''
        img = Image.open(filename)
        img_ = transform(img).unsqueeze(0)
        img_ = img_.to(device)
        outputs = net(img_)
        _, predicted = torch.max(outputs,1)
        pre=predicted.cpu().numpy()
        self.resultDisplay.setText(''.join('%5s' % classes[x] for x in pre))
        '''
        #加载数据集
        filename='MyDataset/test\C1_inclusion\In_1.bmp'
        image=Image.open(filename).convert('RGB') #读取图像，转换为三维矩阵
        classes = ['C1_inclusion','C2_patches','C3_crazing','C4_pitted','C5_rolled-in','C6_scratches']
        with torch.no_grad():
              outputs = net(image)
              _, predicted = torch.max(outputs.data,1)
              pre=predicted.cpu().numpy()
              lab=labels.cpu().numpy()
              self.resultDisplay.setText(''.join('%5s' % classes[x] for x in pre))
            #  QImg = QImage(images, cols, rows, bytesPerLine, QImage.Format_RGB888)
              #self.labelCapture.setPixmap(QPixmap.fromImage(QImg).scaled(
            #  self.labelCapture.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        '''
        

    
    
    

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = PyQtMainEntry()
    window.show()
    sys.exit(app.exec_())
    #python -m PyQt5.uic.pyuic demo01.ui -o demo01.py
    #pyinstaller -F --clean --distpath shark final01.py
