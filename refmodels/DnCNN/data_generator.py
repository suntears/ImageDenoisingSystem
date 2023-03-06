# -*- coding: utf-8 -*-

import glob
import cv2
import numpy as np
from torch.utils.data import Dataset
import torch


patch_size, stride = 40, 10 #补丁大小40 步长10
aug_times = 1
scales = [1, 0.9, 0.8, 0.7]
batch_size = 128

#加噪声类
class DenoisingDataset(Dataset):
    """Dataset wrapping tensors.
    Arguments:
        xs (Tensor): clean image patches
        sigma: noise level, e.g., 25
    """
    def __init__(self, xs, sigma):
        super(DenoisingDataset, self).__init__()
        self.xs = xs
        self.sigma = sigma

    def __getitem__(self, index):
        batch_x = self.xs[index]
        #torch.randn:返回一个Tensor，包含[0,1)的均匀分布的一组随机数，形状由sizes定义
        noise = torch.randn(batch_x.size()).mul_(self.sigma/255.0)
        #添加噪声
        batch_y = batch_x + noise
        #batch_x:干净图像 batch_y:含噪图像
        return batch_y, batch_x

    def __len__(self):
        return self.xs.size(0)#xs.size(0)是batch_size的值


def show(x, title=None, cbar=False, figsize=None):
    import matplotlib.pyplot as plt
    plt.figure(figsize=figsize)
    #interpolation（图像插值）:nearest(最邻近插值) cmap:图像空间，默认RGB(A)
    plt.imshow(x, interpolation='nearest', cmap='gray')
    if title:
        plt.title(title)
    if cbar:
        plt.colorbar()#将颜色条添加到绘图中
    plt.show()

#数据增强
def data_aug(img, mode=0):
    # data augmentation
    if mode == 0:#原图
        return img
    elif mode == 1:#图像上下翻转
        return np.flipud(img)
    elif mode == 2:#图像逆时针旋转90度
        return np.rot90(img)
    elif mode == 3:#图像先逆时针旋转90度，再上下翻转
        return np.flipud(np.rot90(img))
    elif mode == 4:#图像逆时针旋转90*k=180度（k>0逆时针，k<0顺时针)
        return np.rot90(img, k=2)
    elif mode == 5:#图像先逆时针旋转180度，再上下翻转
        return np.flipud(np.rot90(img, k=2))
    elif mode == 6:#图像逆时针旋转90*k=270度
        return np.rot90(img, k=3)
    elif mode == 7:#图像先逆时针旋转90*k=270度，再上下翻转
        return np.flipud(np.rot90(img, k=3))

#从一张图像中获取多尺度的补丁
def gen_patches(file_name):
    # get multiscale patches from a single image
    img = cv2.imread(file_name, 0)  # gray scale
    h, w = img.shape
    patches = []
    for s in scales:
        h_scaled, w_scaled = int(h*s), int(w*s)
        #图像缩放时参数输入是宽x高x通道 INTER_CUBIC:4x4像素邻域的双三次插值 缩小图像
        img_scaled = cv2.resize(img, (h_scaled, w_scaled), interpolation=cv2.INTER_CUBIC)
        # extract patches
        for i in range(0, h_scaled-patch_size+1, stride):
            for j in range(0, w_scaled-patch_size+1, stride):
                x = img_scaled[i:i+patch_size, j:j+patch_size]
                for k in range(0, aug_times):
                    x_aug = data_aug(x, mode=np.random.randint(0, 8))
                    patches.append(x_aug)
    return patches#patches是一个列表

#从数据集中生成干净的补丁
def datagenerator(data_dir='data/train', verbose=False):
    # generate clean patches from a dataset
    #匹配png文件
    file_list = glob.glob(data_dir+'/*.png')  # get name list of all .png files
    # initrialize
    data = []
    # generate patches
    for i in range(len(file_list)):
        patches = gen_patches(file_list[i])
        for patch in patches:    
            data.append(patch)
        if verbose:
            print(str(i+1) + '/' + str(len(file_list)) + ' is done ^_^')
    #转换数据类型
    data = np.array(data, dtype='uint8')
    data = np.expand_dims(data, axis=3)
    discard_n = len(data)-len(data)//batch_size*batch_size  # because of batch namalization
    data = np.delete(data, range(discard_n), axis=0)
    print('^_^-training data finished-^_^')
    return data


if __name__ == '__main__': 
    data = datagenerator(data_dir='data/train')  















































 