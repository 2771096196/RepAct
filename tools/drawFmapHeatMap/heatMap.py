import matplotlib.pyplot as plt
import numpy as np


def plot_heatmap(x, save_path="./save.jpg"):
    """
    plot_heatmap
    param:
    - x: tensor
    - save_path:  save 2 "./save.jpg"
    """
    heatmap_data = x.cpu().detach().numpy()[0]

    plt.figure(figsize=(10, 5))

    for i in range(heatmap_data.shape[0]):
        plt.imshow(heatmap_data[i], cmap='viridis', interpolation='nearest', vmin=0, vmax=2)
        plt.title(f'Heatmap of Channel {i} Output')
        plt.colorbar()
        plt.savefig(save_path.replace(".jpg", f"_channel_{i}.jpg"))
        plt.close()