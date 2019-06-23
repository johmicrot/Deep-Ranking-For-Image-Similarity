import os
import torchvision
import numpy as np
import glob
import cv2
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, RandomHorizontalFlip, RandomVerticalFlip, ToTensor
from config import config as cgf


# Passes an image forward though the network
def forward(x, model, device):
    x = x.type('torch.FloatTensor').to(device)
    return(model(x))


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def cv2_loader(path):
    return cv2.imread(path)


class Tiny(Dataset):
    def __init__(self, root_dir, mode, transform=None, loader = pil_loader):

        if transform is None:
            transform = Compose([Resize(224), RandomHorizontalFlip(p=0.5), RandomVerticalFlip(p=0.5), ToTensor()])
        self.mode = mode
        self.root_dir = root_dir
        self.transform = transform
        self.loader = loader
        # self.class_dict = {}
        self.rev_dict = {}
        self.image_dict = {}
        self.img_class = {}
        labels = []

        if mode == 'train':
            for i, j in enumerate(os.listdir(os.path.join(self.root_dir))):
                # self.class_dict[j] = i
                self.rev_dict[i] = j
                self.image_dict[j] = np.array(os.listdir(os.path.join(self.root_dir, j, 'images')))
                for k, l in enumerate(os.listdir(os.path.join(self.root_dir, j, 'images'))):
                    labels.append((l, i))
            for i, j in enumerate(labels):
                self.img_class[i] = j
            self.num_classes = cgf.NUM_CLASSES

        elif mode == 'model_train':
            for index, im_class in enumerate(os.listdir(os.path.join(self.root_dir, 'train'))):
                # self.class_dict[im_class] = index
                self.rev_dict[index] = im_class
                self.image_dict[im_class] = np.array(os.listdir(os.path.join(self.root_dir, 'train', im_class, 'images')))
                for k, l in enumerate(os.listdir(os.path.join(self.root_dir, 'train', im_class, 'images'))):
                    labels.append((l, index))  # (img name, class)

            for i, j in enumerate(labels):
                self.img_class[i] = j  # (i = img name, j = class)

        elif mode == 'test':
            basedir = '/home/john/Datasets/'
            dataset = basedir + 'tiny-imagenet-%s/' % cgf.NUM_CLASSES
            current_classes = glob.glob(dataset + '/train/*')
            for i in range(len(current_classes)):
                current_classes[i] = current_classes[i].split('/')[-1]
            val = '/home/john/Datasets/tiny-imagenet-%s/val_annotations.txt' % cgf.NUM_CLASSES
            df = pd.read_csv(val, sep='\t', header=None, names=['image', 'im_class', '1', '2', '3', '4'])
            df2 = df[df.im_class.isin(current_classes)]
            self.img_class = list(df2.image)
            self.image_class = np.array(df2[['image', 'im_class']]).astype('str')
            self.class_dic = {}
            for i in self.image_class:
                self.class_dic[i[0]] = i[1]

    def _sample(self, idx):
        if self.mode == 'model_train':
            im, im_class = self.img_class[idx]
            im2 = np.random.choice(self.image_dict[self.rev_dict[im_class]])
            numbers = list(range(0, im_class)) + list(range(im_class + 1, cgf.NUM_CLASSES))
            class3 = np.random.choice(numbers)
            im3 = np.random.choice(self.image_dict[self.rev_dict[class3]])
            q = os.path.join(self.root_dir, 'train', self.rev_dict[im_class], 'images', im)
            pos = os.path.join(self.root_dir, 'train', self.rev_dict[im_class], 'images', im2)
            neg = os.path.join(self.root_dir, 'train', self.rev_dict[class3], 'images', im3)
            return [q, pos, neg]

        elif self.mode == 'test':
            path = os.path.join(self.root_dir, self.img_class[idx])
            return path, self.class_dic[self.img_class[idx]]
        elif self.mode == 'train':
            im, im_class = self.img_class[idx]
            path = os.path.join(self.root_dir,self.rev_dict[im_class],'images', im)
            return path, im_class

    def __len__(self):
        if self.mode == 'model_train':
            return cgf.NUM_CLASSES * 500
        else:
            return len(self.img_class)

    def __getitem__(self, idx):
        if self.mode == 'model_train':
            paths = self._sample(idx)
            images = []
            for i in paths:
                img = self.loader(i)
                if self.transform:
                    # line below commented out due to memory overflow, so no augmentation is performed
                    # temp = self.transform(temp)
                    t = np.array(img)
                    img = np.transpose(t, (2, 0, 1))
                else:
                    img = np.array(img)
                    img = np.transpose(img, (2, 0, 1))
                images.append(img)
            return images[0], images[1], images[2]
        else:
            paths, im_class = self._sample(idx)
            img = self.loader(paths)
            if self.transform:
                img = self.transform(img)
            return img, im_class

