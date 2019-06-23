import numpy as np
from sklearn.neighbors import NearestNeighbors
from config import config as cfg


def f_train(x):
    return train_labels[x]

def f_test(x):
    return test_labels[x]


NN = NearestNeighbors(n_neighbors=cfg.NUM_CLASSES)


# Calculate train accuracy
train_latents = np.load(cfg.train_save_dir)
train_labels = np.load(cfg.train_save_labels_dir)
NN.fit(train_latents, train_labels)
_,ind = NN.kneighbors(train_latents)
ind = ind.ravel()
# ind2 = np.array(list(map(f,ind))).reshape(100, cfg.NUM_CLASSES)
ind_class = np.array(list(map(f_train, ind)))
class_comparison = ind_class == train_labels.reshape(-1, 1)
print('The train accuracy obtained is ')
print(sum(class_comparison.mean(axis=1)) / len(train_labels))

# Calculate train accuracy
test_latents = np.load(cfg.test_save_dir)
test_labels = np.load(cfg.test_save_labels_dir)
NN.fit(test_latents, test_labels)
_,ind = NN.kneighbors(test_latents)
ind = ind.ravel()
ind_class = np.array(list(map(f_test, ind)))
class_comparison = ind_class == test_labels.reshape(-1, 1)
print('The test accuracy obtained is ')
print(sum(class_comparison.mean(axis=1)) / len(test_labels))