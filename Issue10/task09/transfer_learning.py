# -*- coding:utf-8 -*-
import time
import torch
import torch.nn.functional as F
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s: %(message)s')


def preprocess(PIL_img, image_shape):
    process = torchvision.transforms.Compose([
        torchvision.transforms.Resize(image_shape),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=rgb_mean, std=rgb_std)])

    return process(PIL_img).unsqueeze(dim=0)  # (batch_size, 3, H, W)


def postprocess(img_tensor):
    inv_normalize = torchvision.transforms.Normalize(
        mean=-rgb_mean / rgb_std,
        std=1/rgb_std)
    to_PIL_image = torchvision.transforms.ToPILImage()
    return to_PIL_image(inv_normalize(img_tensor[0].cpu()).clamp(0, 1))


def extract_features(X, content_layers, style_layers):
    ''' 逐层进行前向传播 '''
    contents = []
    styles = []
    for i in range(len(net)):
        X = net[i](X)
        if i in style_layers:
            styles.append(X)
        if i in content_layers:
            contents.append(X)
    return contents, styles


def get_contents(image_shape, device):
    ''' 提取内容图像特征 '''
    content_X = preprocess(content_img, image_shape).to(device)
    contents_Y, _ = extract_features(content_X, content_layers, style_layers)
    return content_X, contents_Y


def get_styles(image_shape, device):
    ''' 提取样式图像特征 '''
    style_X = preprocess(style_img, image_shape).to(device)
    _, styles_Y = extract_features(style_X, content_layers, style_layers)
    return style_X, styles_Y


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'use device: {device}, \t version of torch: {torch.__version__}')

    rgb_mean = np.array([0.485, 0.456, 0.406])
    rgb_std = np.array([0.229, 0.224, 0.225])

    # 读取内容图像和样式图像
    content_img = Image.open('./data/NeuralStyle/rainier.jpg')
    style_img = Image.open('./data/NeuralStyle/autumn_oak.jpg')

    # 加载预训练模型
    pretrained_net = torchvision.models.vgg19(pretrained=False)
    pretrained_net.load_state_dict(torch.load('./data/vgg/vgg19-dcbb9e9d.pth'))

    # 选取样式层和内容层
    style_layers, content_layers = [0, 5, 10, 19, 28], [25]
    # 抽取需要用到的层
    net_list = []
    for i in range(max(content_layers + style_layers) + 1):
        net_list.append(pretrained_net.features[i])
    net = torch.nn.Sequential(*net_list)
