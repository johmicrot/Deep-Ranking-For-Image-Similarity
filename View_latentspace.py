from sklearn.decomposition import PCA
from config import config as cfg
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.cm as cm

def plot_scatter(values, cls, title):
    # Create a color-map with a different color for each class.
    cmap = cm.rainbow(np.linspace(0.0, 1.0, cfg.NUM_CLASSES))

    # Get the color for each sample.
    colors = cmap[cls]

    # Extract the x- and y-values.
    x = values[:, 0]
    y = values[:, 1]
    plt.title(title)
    plt.scatter(x, y, color=colors, marker='.')
    plt.show()

# Calclate and display the PCA in two dimentions
train_latent = np.load(cfg.train_save_dir)
labels = np.load(cfg.train_save_labels_dir)
pca = PCA(n_components=100) #1000 is overkill but it's quick
pca_transfer_values = pca.fit_transform(train_latent)
plot_scatter(pca_transfer_values, labels, '200 Train Dataset. Latent space with PCA 100 componets')
print('Now calculate and display the TSNE in 2 dimentions')
# Calculate and display the TSNE in 2 dimentions
tsne = TSNE(n_components=2)
transfer_values_50d = tsne.fit_transform(train_latent)
plot_scatter(pca_transfer_values, labels, 'Train Dataset. Latent space with t-SNE')