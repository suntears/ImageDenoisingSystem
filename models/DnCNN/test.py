# -*- coding: utf-8 -*-

import argparse
import os, time, datetime
import numpy as np
import torch.nn as nn
import torch.nn.init as init
import torch
#计算两幅图像的峰值信噪比PSNR和平均结构相似性指数SSIM
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from skimage.io import imread, imsave


def parse_args():
    parser = argparse.ArgumentParser()
    #测试集 data/test
    parser.add_argument('--set_dir', default='.\\data\\test', type=str, help='directory of test dataset')
    #测试集名字 set68 set12
    parser.add_argument('--set_names', default=['Set12','Set68'], help='directory of test dataset')
    #噪声水平 25
    parser.add_argument('--sigma', default=25, type=int, help='noise level')
    #模型路径 models/DnCNN_sigma25
    parser.add_argument('--model_dir', default=os.path.join('models', 'DnCNN_sigma25'), help='directory of the model')
    #模型名字 model_001.pth
    parser.add_argument('--model_name', default='model_002.pth', type=str, help='the model name')
    #结果位置 results
    parser.add_argument('--result_dir', default='results', type=str, help='directory of test dataset')
    #是否保存结果 1保存，0不保存
    parser.add_argument('--save_result', default=1, type=int, help='save the denoised image, 1 or 0')
    return parser.parse_args()


def log(*args, **kwargs):
     print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), *args, **kwargs)


def save_result(result, path):
    path = path if path.find('.') != -1 else path+'.png'
    ext = os.path.splitext(path)[-1]
    if ext in ('.txt', '.dlm'):
        np.savetxt(path, result, fmt='%2.4f')
    else:
        imsave(path, np.clip(result, 0, 1))


def show(x, title=None, cbar=False, figsize=None):
    import matplotlib.pyplot as plt
    plt.figure(figsize=figsize)
    plt.imshow(x, interpolation='nearest', cmap='gray')
    if title:
        plt.title(title)
    if cbar:
        plt.colorbar()
    plt.show()


class DnCNN(nn.Module):

    def __init__(self, depth=17, n_channels=64, image_channels=1, use_bnorm=True, kernel_size=3):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        layers = []
        layers.append(nn.Conv2d(in_channels=image_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=True))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(depth-2):
            layers.append(nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(n_channels, eps=0.0001, momentum=0.95))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=n_channels, out_channels=image_channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)
        self._initialize_weights()

    def forward(self, x):
        y = x
        out = self.dncnn(x)
        return y-out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                print('init weight')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)


if __name__ == '__main__':

    args = parse_args()

    # model = DnCNN()
    if not os.path.exists(os.path.join(args.model_dir, args.model_name)):

        model = torch.load(os.path.join(args.model_dir, 'model.pth'))
        # load weights into new model
        log('load trained model on Train400 dataset by kai')
    else:
        # model.load_state_dict(torch.load(os.path.join(args.model_dir, args.model_name)))
        model = torch.load(os.path.join(args.model_dir, args.model_name))
        log('load trained model')

    #在模型测试时使用，关闭BN层和Dropout层
    model.eval()  # evaluation mode

    if torch.cuda.is_available():
        model = model.cuda()

    if not os.path.exists(args.result_dir):
        os.mkdir(args.result_dir)

    for set_cur in args.set_names:

        if not os.path.exists(os.path.join(args.result_dir, set_cur)):
            os.mkdir(os.path.join(args.result_dir, set_cur))
        psnrs = []#峰值信噪比
        ssims = []#结构相似性

        for im in os.listdir(os.path.join(args.set_dir, set_cur)):
            if im.endswith(".jpg") or im.endswith(".bmp") or im.endswith(".png"):
                #np.array将读入图像转为ndarray形式，使像素值为[0,1]
                x = np.array(imread(os.path.join(args.set_dir, set_cur, im)), dtype=np.float32)/255.0
                np.random.seed(seed=0)  # for reproducibility
                #添加高斯噪声
                y = x + np.random.normal(0, args.sigma/255.0, x.shape)  # Add Gaussian noise without clipping
                y = y.astype(np.float32)
                #将数据转为torch张量
                y_ = torch.from_numpy(y).view(1, -1, y.shape[0], y.shape[1])
                #等待所有操作完成，同步时间
                torch.cuda.synchronize()
                start_time = time.time()
                y_ = y_.cuda()
                x_ = model(y_)  # inference
                x_ = x_.view(y.shape[0], y.shape[1])
                #从gpu数据转为cpu数据
                x_ = x_.cpu()
                x_ = x_.detach().numpy().astype(np.float32)
                
                torch.cuda.synchronize()
                elapsed_time = time.time() - start_time
                print('setName: %10s ,image: %10s ,time: %2.4f second' % (set_cur, im, elapsed_time))
                
                #计算PSNR和SSIM
                #psnr_x_ = compare_psnr(x, x_)
                #ssim_x_ = compare_ssim(x, x_)
                psnr_x_ = compare_psnr(x, y-x_)
                ssim_x_ = compare_ssim(x, y-x_)
                if args.save_result:
                    #分离文件名和扩展名
                    name, ext = os.path.splitext(im)
                    #show(np.hstack((y, x_)))  # show the image
                    #save_result(x_, path=os.path.join(args.result_dir, set_cur, name+'_dncnn'+ext))  # save the denoised image

                    show(np.hstack((y, y-x_)))  # show the image
                    save_result(y-x_, path=os.path.join(args.result_dir, set_cur, name+'_dncnn'+ext))  # save the denoised image
                psnrs.append(psnr_x_)
                ssims.append(ssim_x_)
        psnr_avg = np.mean(psnrs)
        ssim_avg = np.mean(ssims)
        psnrs.append(psnr_avg)
        ssims.append(ssim_avg)
        if args.save_result:
            save_result(np.hstack((psnrs, ssims)), path=os.path.join(args.result_dir, set_cur, 'results.txt'))
        log('Datset: {0:10s} \n  PSNR = {1:2.2f}dB, SSIM = {2:1.4f}'.format(set_cur, psnr_avg, ssim_avg))







































