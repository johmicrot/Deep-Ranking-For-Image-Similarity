import torch
import numpy as np
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
from Data_handler import Tiny
from config import config as cfg

# Performs a forward pass through the network
def forward(x):
    x = x.type('torch.FloatTensor').to(device)
    return(model(x))


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create the model
model_location = 'model/%sclasses/MRS7.ckpt' % cfg.NUM_CLASSES
model = models.resnet50(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, cfg.NUM_FEATURES)
model = model.to(device)
model.load_state_dict(torch.load(model_location))


# Create DataLoader for test
test_set = Tiny(root_dir=cfg.dataset_dir + '/val/images', mode='test')
dataloader_test = DataLoader(test_set, batch_size=100)

print('Generate test set latent features')
# initialize the embeddings or latent space
embedding = torch.randn(0, cfg.NUM_FEATURES).type('torch.FloatTensor').to(device)
model.eval()
with torch.no_grad():
    labels = []
    for k, (i, j) in enumerate(dataloader_test):
        print(k, end='\r')
        latent_space = forward(i)
        embedding = torch.cat((embedding, latent_space), 0)
        labels = labels + list(j.numpy())
eb = embedding.cpu().numpy()
print('Saving test latent and labels to %s' % cfg.train_out_dir)
np.save(cfg.test_save_dir, eb)
np.save(cfg.test_save_labels_dir, labels)

# just a security, might not be necessary
del test_set
del dataloader_test


# Create DataLoader for train
train_set = Tiny(root_dir=cfg.dataset_dir + '/train', mode='train')
dataloader_train = DataLoader(train_set, batch_size=100)


print('Generate train set latent features')
# initialize the embeddings or latent space
embedding = torch.randn(0, cfg.NUM_FEATURES).type('torch.FloatTensor').to(device)
model.eval()
with torch.no_grad():
    labels = []
    for k, (i, j) in enumerate(dataloader_train):
        print(k, end='\r')
        latent_space = forward(i)
        embedding = torch.cat((embedding, latent_space), 0)
        labels = labels + list(j.numpy())
eb = embedding.cpu().numpy()
print('Saving train latent and labels to %s' % cfg.train_out_dir)
np.save(cfg.train_save_dir, eb)
np.save(cfg.train_save_labels_dir, labels)

