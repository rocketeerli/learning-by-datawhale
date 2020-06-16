import torchvision
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


class Dataset():
    def __init__(self, batch_size=256, resize=None, root='./data', num_workers=4):
        self.classes = [
            't-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal',
            'shirt', 'sneaker', 'bag', 'ankle boot'
        ]
        self.batch_size = batch_size
        trans = []
        if resize:
            trans.append(torchvision.transforms.Resize(size=resize))
        trans.append(torchvision.transforms.ToTensor())
        transform = torchvision.transforms.Compose(trans)
        self.mnist_train = torchvision.datasets.FashionMNIST(
            root=root,
            train=True,
            download=True,
            transform=transform)
        self.mnist_test = torchvision.datasets.FashionMNIST(
            root=root,
            train=False,
            download=True,
            transform=transform)
        self.train_iter = DataLoader(self.mnist_train,
                                     batch_size=batch_size,
                                     shuffle=True,
                                     num_workers=num_workers)
        self.test_iter = DataLoader(self.mnist_test,
                                    batch_size=batch_size,
                                    shuffle=False,
                                    num_workers=num_workers)

    def get_labels(self, labels):
        return [self.classes[int(i)] for i in labels]

    def show_data(self, images, labels):
        labels = self.get_labels(labels)
        _, figs = plt.subplots(1, len(images), figsize=(12, 12))
        for f, img, lbl in zip(figs, images, labels):
            f.imshow(img.view((28, 28)).numpy())
            f.set_title(lbl)
            f.axes.get_xaxis().set_visible(False)
            f.axes.get_yaxis().set_visible(False)
        plt.show()
