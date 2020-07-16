# -*- coding:utf-8 -*-
import torch
from PIL import Image
import matplotlib.pyplot as plt
from utils import MultiBoxPrior, bbox_to_rect, show_bboxes, compute_jaccard, MultiBoxTarget, MultiBoxDetection
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s: %(message)s')


def test_read_daa(img):
    dog_bbox, cat_bbox = [60, 45, 378, 516], [400, 112, 655, 493]
    fig = plt.imshow(img)
    fig.axes.add_patch(bbox_to_rect(dog_bbox, 'blue'))
    fig.axes.add_patch(bbox_to_rect(cat_bbox, 'red'))
    plt.show()


def test_anchors(img):
    logging.info('选出每个像素点的 anchors 并展示其中一个')
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

    logging.info('展示 groundtruth 和 anchor box')
    bbox_scale = torch.tensor((w, h, w, h), dtype=torch.float32)
    ground_truth = torch.tensor([[0, 0.1, 0.08, 0.52, 0.92],
                                [1, 0.55, 0.2, 0.9, 0.88]])
    anchors = torch.tensor([[0, 0.1, 0.2, 0.3], [0.15, 0.2, 0.4, 0.4],
                            [0.63, 0.05, 0.88, 0.98], [0.66, 0.45, 0.8, 0.8],
                            [0.57, 0.3, 0.92, 0.9]])
    fig = plt.imshow(img)
    show_bboxes(fig.axes, ground_truth[:, 1:] * bbox_scale, ['dog', 'cat'], 'k')
    show_bboxes(fig.axes, anchors * bbox_scale, ['0', '1', '2', '3', '4'])
    plt.show()
    iou = compute_jaccard(anchors, ground_truth[:, 1:]) # 验证一下写的compute_jaccard函数
    logging.info(f'iou: {iou}')

    labels = MultiBoxTarget(anchors.unsqueeze(dim=0),
                            ground_truth.unsqueeze(dim=0))
    logging.info(f'label cls: {labels[2]}')

    logging.info('输出边界框 测试 NMS')
    anchors = torch.tensor([[0.1, 0.08, 0.52, 0.92], [0.08, 0.2, 0.56, 0.95],
                            [0.15, 0.3, 0.62, 0.91], [0.55, 0.2, 0.9, 0.88]])
    offset_preds = torch.tensor([0.0] * (4 * len(anchors)))
    cls_probs = torch.tensor([[0., 0., 0., 0.],  # 背景的预测概率
                              [0.9, 0.8, 0.7, 0.1],  # 狗的预测概率
                              [0.1, 0.2, 0.3, 0.9]])  # 猫的预测概率
    fig = plt.imshow(img)
    show_bboxes(fig.axes, anchors * bbox_scale,
                ['dog=0.9', 'dog=0.8', 'dog=0.7', 'cat=0.9'])
    plt.show()
    output = MultiBoxDetection(
                cls_probs.unsqueeze(dim=0), offset_preds.unsqueeze(dim=0),
                anchors.unsqueeze(dim=0), nms_threshold=0.5)
    logging.info(f'output: {output}')
    fig = plt.imshow(img)
    for i in output[0].detach().cpu().numpy():
        if i[0] == -1:
            continue
        label = ('dog=', 'cat=')[int(i[0])] + str(i[1])
        show_bboxes(fig.axes, [torch.tensor(i[2:]) * bbox_scale], label)
    plt.show()


if __name__ == '__main__':
    img = Image.open('./data/img2083/img/catdog.jpg')
    w, h = img.size
    logging.info(f'w of image: {w} \t h of image: {h}')

    # test_read_daa(img)

    test_anchors(img)
