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


def content_loss(Y_hat, Y):
    ''' 内容损失 '''
    return F.mse_loss(Y_hat, Y)


def gram(X):
    ''' 计算特征图的格莱姆矩阵 '''
    num_channels, n = X.shape[1], X.shape[2] * X.shape[3]
    X = X.view(num_channels, n)
    return torch.matmul(X, X.t()) / (num_channels * n)


def style_loss(Y_hat, gram_Y):
    ''' 样式损失 '''
    return F.mse_loss(gram(Y_hat), gram_Y)


def tv_loss(Y_hat):
    ''' 总变差损失 '''
    return 0.5 * (F.l1_loss(Y_hat[:, :, 1:, :], Y_hat[:, :, :-1, :]) +
                  F.l1_loss(Y_hat[:, :, :, 1:], Y_hat[:, :, :, :-1]))


def compute_loss(X, contents_Y_hat, styles_Y_hat, contents_Y, styles_Y_gram):
    ''' 分别计算内容损失、样式损失和总变差损失 '''
    contents_l = [content_loss(Y_hat, Y) * content_weight for Y_hat, Y in zip(
        contents_Y_hat, contents_Y)]
    styles_l = [style_loss(Y_hat, Y) * style_weight for Y_hat, Y in zip(
        styles_Y_hat, styles_Y_gram)]
    tv_l = tv_loss(X) * tv_weight
    # 对所有损失求和
    l = sum(styles_l) + sum(contents_l) + tv_l
    return contents_l, styles_l, tv_l, l


class GeneratedImage(torch.nn.Module):
    def __init__(self, img_shape):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.rand(*img_shape))

    def forward(self):
        return self.weight


def get_inits(X, device, lr, styles_Y):
    gen_img = GeneratedImage(X.shape).to(device)
    gen_img.weight.data = X.data
    optimizer = torch.optim.Adam(gen_img.parameters(), lr=lr)
    styles_Y_gram = [gram(Y) for Y in styles_Y]
    return gen_img(), styles_Y_gram, optimizer


def train(X, contents_Y, styles_Y, device, lr, max_epochs, lr_decay_epoch):
    print("training on ", device)
    X, styles_Y_gram, optimizer = get_inits(X, device, lr, styles_Y)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lr_decay_epoch, gamma=0.1)
    for i in range(max_epochs):
        start = time.time()

        contents_Y_hat, styles_Y_hat = extract_features(
                X, content_layers, style_layers)
        contents_l, styles_l, tv_l, l = compute_loss(
                X, contents_Y_hat, styles_Y_hat, contents_Y, styles_Y_gram)

        optimizer.zero_grad()
        l.backward(retain_graph=True)
        optimizer.step()
        scheduler.step()

        if i % 50 == 0 and i != 0:
            print('epoch %3d, content loss %.2f, style loss %.2f, '
                  'TV loss %.2f, %.2f sec'
                  % (i, sum(contents_l).item(), sum(styles_l).item(), tv_l.item(),
                     time.time() - start))
    return X.detach()


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

    # 损失函数权值
    content_weight, style_weight, tv_weight = 1, 1e3, 10

    # 开始训练
    image_shape = (150, 225)
    net = net.to(device)
    content_X, contents_Y = get_contents(image_shape, device)
    style_X, styles_Y = get_styles(image_shape, device)
    output = train(content_X, contents_Y, styles_Y, device, 0.01, 500, 200)

    plt.imshow(postprocess(output))
    plt.show()

    # 在更大的尺寸上训练
    image_shape = (300, 450)
    _, content_Y = get_contents(image_shape, device)
    _, style_Y = get_styles(image_shape, device)
    X = preprocess(postprocess(output), image_shape).to(device)
    big_output = train(X, content_Y, style_Y, device, 0.01, 500, 200)

    plt.imshow(postprocess(big_output))
    plt.show()
