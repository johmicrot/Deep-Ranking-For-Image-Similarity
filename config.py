import os
from easydict import EasyDict as edict

config = edict()

config.NUM_CLASSES = 2
config.NUM_FEATURES = 2000
config.BATCH_SIZE =  18
config.NUM_EPOCHS = 60
config.LEARNING_RATE = 0.000005
config.START_EPOCH = 0
config.AUGMENT = False
# margin for the triplet hinge loss
config.MARGIN = 1

#number of images to show during the query process
config.IMAGES_SHOW = 5

config.basedir = '/home/john/Datasets/'
config.dataset_dir = config.basedir + 'tiny-imagenet-%s/' % config.NUM_CLASSES

config.train_out_dir = 'train_latent/%sclasses' % config.NUM_CLASSES
config.train_save_dir = '%s/train_latents.npy' % config.train_out_dir
config.train_save_labels_dir = '%s/train_labels.npy' % config.train_out_dir

config.test_out_dir = 'test_latent/%sclasses' % config.NUM_CLASSES
config.test_save_dir = '%s/test_latents.npy' % config.test_out_dir
config.test_save_labels_dir = '%s/test_labels.npy' % config.test_out_dir

if not os.path.exists(config.test_out_dir):
    os.makedirs(config.test_out_dir)
if not os.path.exists(config.train_out_dir):
    os.makedirs(config.train_out_dir )

config.out_model_dir = 'model/%sclasses' % config.NUM_CLASSES
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
