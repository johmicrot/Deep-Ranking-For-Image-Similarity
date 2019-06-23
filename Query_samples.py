import torch
import numpy as np
from PIL import Image
import glob
import pandas as pd
import cv2
from config import config as cfg
from Data_handler import Tiny


'''# Here we are isolating only the correct valition images from the dataset.
# The dataset has all 200 classes in one folder so this is one way to
# seperate them out
current_classes = glob.glob(cfg.dataset_dir + '/train/*')
for i in range(len(current_classes)):
    current_classes[i] = current_classes[i].split('/')[-1]
val = cfg.dataset_dir + '/val_annotations.txt'
df = pd.read_csv(val, sep='\t', header=None, names=['image', 'im_class', 'None', 'None', 'None', 'None'])
df2 = df[df.im_class.isin(current_classes)]
'''
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load datasets and latent features
test_dataset = Tiny(root_dir=cfg.dataset_dir + '/val/images', mode='test')
train_dataset = Tiny(root_dir=cfg.dataset_dir + '/train', mode='train')
train_latents = np.load(cfg.train_save_dir)
test_latents = np.load(cfg.test_save_dir)

len_train = len(train_latents)
for i in range(len_train):
    idx = np.random.randint(len_train)
    idx = i
    train_latent = train_latents[idx]
    print('Query image used is:')
    query,_ = train_dataset._sample(idx)
    img = cv2.imread(query)


    print(train_dataset._sample(idx))

    root_sqr_err = (train_latents - train_latent) ** 2
    root_sqr_err = root_sqr_err.sum(axis=1)
    print((np.sort(root_sqr_err) ** 0.5)[:5])
    idx_closest_neighbor = root_sqr_err.argsort()[:5]
    print('Closest images are:')
    for j in idx_closest_neighbor:
        print(train_dataset._sample(j)[0])
        comparison_img = train_dataset._sample(j)[0]
        comparison = cv2.imread(comparison_img)
        img = cv2.hconcat((img, comparison))

    shape = img.shape
    img = cv2.resize(img, (shape[1]*3, shape[0]*3))

    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
