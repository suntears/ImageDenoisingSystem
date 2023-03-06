# -*- coding: utf-8 -*-


import argparse#Python参数解析模块
import re#Python正则表达式模块
import os, glob, datetime, time#文件名操作模块glob
import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
import torch.nn.init as init
from torch.utils.data import DataLoader#实现数据以什么方式输入到什么网络中
import torch.optim as optim#优化算法库
from torch.optim.lr_scheduler import MultiStepLR#根据迭代来调整学习率的算法
import data_generator as dg#导入处理数据文件
from data_generator import DenoisingDataset


# Params
#ArgumentParser对象包含将命令行解析成Python数据类型所需的全部信息
parser = argparse.ArgumentParser(description='PyTorch DnCNN')
#add_argument函数指定ArgumentParser如何获取命令行字符串，并将其转换为对象
#模型 字符串 默认DnCNN
parser.add_argument('--model', default='DnCNN', type=str, help='choose a type of model')
#批量大小 整型 默认128
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
#训练数据路径 字符串 默认data/train
parser.add_argument('--train_data', default='.\\data\\train', type=str, help='path of train data')
#噪声水平 整型 默认25
parser.add_argument('--sigma', default=25, type=int, help='noise level')
#迭代轮数 整型 默认180
parser.add_argument('--epoch', default=180, type=int, help='number of train epoches')
#学习率 浮点数 默认0.001 adam优化算法
parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate for Adam')
#parse_args函数检查命令行，把每个参数转换为适当的类型，然后调用相应的操作
args = parser.parse_args()

#args传递参数
batch_size = args.batch_size
n_epoch = args.epoch
sigma = args.sigma
#判断cuda是否可用
cuda = torch.cuda.is_available()

#os.path.join:拼接文件路径 save_dir=models/DNCNN_sigma25 保存训练好的模型
save_dir = os.path.join('models', args.model+'_' + 'sigma' + str(sigma))
#判断路径是否存在，不存在则新建路径
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

#定义模型类DnCNN
class DnCNN(nn.Module):
    #初始化
    def __init__(self, depth=17, n_channels=64, image_channels=1, use_bnorm=True, kernel_size=3):
        super(DnCNN, self).__init__()
        #自定义网络模型
        kernel_size = 3#卷积核的大小
        padding = 1#padding表示图片周围填充0的多少。padding=1表示四周都填充0
        
        layers = []
        #四个参数：输入通道数 输出通道数 卷积核大小 padding bias
        #(bias默认为True，但是如果Conv2d后面接BN层的时候，令bias=False。因为此时加不加bias没有影响，加了还占显卡内存
        
        #构建一个输入通道为channels，输出通道为64，卷积核大小为3x3，四周进行1像素点的零填充的conv1层
        layers.append(nn.Conv2d(in_channels=image_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=True))
        #添加非线性激活函数，一般放在在卷积层或BN层之后，池化层之前。inplace=True则会把输出直接覆盖到输入，可以节省显存
        layers.append(nn.ReLU(inplace=True))
        for _ in range(depth-2):
            #构建卷积层
            layers.append(nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=False))
            #加快收敛速度——BN层。输入通道为64，与卷积层输出通道数对应
            #eps是为保证数值稳定性，给分母加上的值（分母不能太接近零）。eps默认为1e-4
            #momentum：动态均值和动态方差所使用的动量，默认0.1
            layers.append(nn.BatchNorm2d(n_channels, eps=0.0001, momentum = 0.95))
            layers.append(nn.ReLU(inplace=True))
        #构建卷积层
        layers.append(nn.Conv2d(in_channels=n_channels, out_channels=image_channels, kernel_size=kernel_size, padding=padding, bias=False))
        #利用nn.Sequential()按顺序构建网络
        self.dncnn = nn.Sequential(*layers)
        #调用初始化权重函数
        self._initialize_weights()

    #定义自己的前向传播函数
    def forward(self, x):
        y = x
        out = self.dncnn(x)
        return y-out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                #正交初始化，主要用于解决深度网络下梯度消失、梯度爆炸问题
                init.orthogonal_(m.weight)
                print('init weight')
                if m.bias is not None:
                    #常数初始化
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                #常数初始化
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

#损失函数类
class sum_squared_error(_Loss):  # PyTorch 0.4.1
    """
    Definition: sum_squared_error = 1/2 * nn.MSELoss(reduction = 'sum')
    The backward is defined as: input-target
    """
    def __init__(self, size_average=None, reduce=None, reduction='sum'):
        super(sum_squared_error, self).__init__(size_average, reduce, reduction)

    def forward(self, input, target):
        return torch.nn.functional.mse_loss(input, target, size_average=None, reduce=None, reduction='sum').div_(2)

#返回之前训练的最后一轮模型（最大轮数），没有则返回0
def findLastCheckpoint(save_dir):
    #返回所有匹配的文件路径列表
    file_list = glob.glob(os.path.join(save_dir, 'model_*.pth'))
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            result = re.findall(".*model_(.*).pth.*", file_)
            epochs_exist.append(int(result[0]))
        initial_epoch = max(epochs_exist)
    else:
        initial_epoch = 0
    return initial_epoch


def log(*args, **kwargs):
     print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), *args, **kwargs)


if __name__=="__main__":
    #建立模型
    print('===> Building model')
    model = DnCNN()
    #从上次训练结束的地方继续训练
    initial_epoch = findLastCheckpoint(save_dir=save_dir)  # load the last model in matconvnet style
    if initial_epoch > 0:
        print('resuming by loading epoch %03d' % initial_epoch)
        # model.load_state_dict(torch.load(os.path.join(save_dir, 'model_%03d.pth' % initial_epoch)))
        model = torch.load(os.path.join(save_dir, 'model_%03d.pth' % initial_epoch))
    
    #启用BatchNormalization和Dropout,如果模型中有BN层和Dropout层，需要调用train函数
    model.train()
    #损失函数
    criterion = sum_squared_error()
    if cuda:
        model = model.cuda()
    #Optimizer 采用Adam算法优化
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    #milestones是一个数组，gamma是0.1的倍数，用于调整学习率
    scheduler = MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.2)  # learning rates
    
    for epoch in range(initial_epoch, n_epoch):
        #调整本轮学习率
        scheduler.step(epoch)  # step to the learning rate in this epcoh
        #生成训练数据
        xs = dg.datagenerator(data_dir=args.train_data)
        #对数据处理，将数据区间变为[0,1]
        xs = xs.astype('float32')/255.0
        #将numpy.ndarray转为pytorch的Tensor，并转为NCHW。
        #N:图像数量，一般为Batch，C：图像通道（灰色为1，彩色为3），H、W：图像宽高（像素）
        xs = torch.from_numpy(xs.transpose((0, 3, 1, 2)))  # tensor of the clean patches, NXCXHXW
        #给图像添加噪声
        DDataset = DenoisingDataset(xs, sigma)
        #num_workers:使用多少个子进程导入数据
        #drop_last:丢弃最后数据，默认False。因为设置batch_size后，最后一批数据可能不是恰好等于batch_size，可以通过这个参数选择是否丢弃这部分数据
        #shuffle:默认为False。设定每次迭代训练的时候是否打乱数据的顺序。
        DLoader = DataLoader(dataset=DDataset, num_workers=4, drop_last=True, batch_size=batch_size, shuffle=True)
        epoch_loss = 0
        start_time = time.time()

        for n_count, batch_yx in enumerate(DLoader):
                optimizer.zero_grad()#梯度置零
                #转为cuda数据形式
                if cuda:
                    batch_x, batch_y = batch_yx[1].cuda(), batch_yx[0].cuda()
                
                #loss = criterion(model(batch_y), batch_x)#计算loss
                loss = criterion(batch_y-model(batch_y), batch_x)#计算loss
                epoch_loss += loss.item()#对loss求和
                #反向传播
                loss.backward()
                #adam优化
                optimizer.step()
                #每处理10张图片，输出1次提示信息
                if n_count % 10 == 0:
                    print('epoch%4d: %4d / %4d loss = %2.4f' % (epoch+1, n_count, xs.size(0)//batch_size, loss.item()/batch_size))
        elapsed_time = time.time() - start_time

        log('epoch = %4d , loss = %4.4f , time = %4.2f s' % (epoch+1, epoch_loss/n_count, elapsed_time))
        
        np.savetxt('train_result.txt', np.hstack((epoch+1, epoch_loss/n_count, elapsed_time)), fmt='%2.4f')
        #保存模型
        # torch.save(model.state_dict(), os.path.join(save_dir, 'model_%03d.pth' % (epoch+1)))
        torch.save(model, os.path.join(save_dir, 'model_%03d.pth' % (epoch+1)))

















































