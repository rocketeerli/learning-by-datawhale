# -*- coding:utf-8 -*-
import torch
from PIL import Image
import matplotlib.pyplot as plt
from utils import MultiBoxPrior, bbox_to_rect, show_bboxes
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s: %(message)s')


def test_read_daa(img):
    dog_bbox, cat_bbox = [60, 45, 378, 516], [400, 112, 655, 493]
    fig = plt.imshow(img)
    fig.axes.add_patch(bbox_to_rect(dog_bbox, 'blue'))
    fig.axes.add_patch(bbox_to_rect(cat_bbox, 'red'))
    plt.show()


if __name__ == '__main__':
    img = Image.open('./data/img2083/img/catdog.jpg')
    w, h = img.size
    logging.info(f'w of image: {w} \t h of image: {h}')
    # test_read_daa(img)

    X = torch.Tensor(1, 3, h, w)  # 构造输入数据
    Y = MultiBoxPrior(X, sizes=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5])
    logging.info(f'shape : {Y.shape}')

    # 展示某个像素点的anchor
    boxes = Y.reshape((h, w, 5, 4))
    fig = plt.imshow(img)
    bbox_scale = torch.tensor([[w, h, w, h]], dtype=torch.float32)
    show_bboxes(fig.axes, boxes[250, 250, :, :] * bbox_scale,
                ['s=0.75, r=1', 's=0.75, r=2', 's=0.75, r=0.5', 's=0.5, r=1', 's=0.25, r=1'])
    plt.show()
