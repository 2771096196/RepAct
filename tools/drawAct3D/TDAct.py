import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def plot_activation_functions_custom_z(activation_functions, num_epoch_list, z=""):
    x_range = np.linspace(-6, 6, 100)
    plt.style.use('seaborn-white')  
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    z_str = 'Activation Functions Lines with ' + z + ' in 3D'
    ax.set_title(z_str)
    PlotLen = len(num_epoch_list)
    for epoch, activation_fn in enumerate(activation_functions):
        activation_fn.to("cuda")
        x_tensor = torch.tensor(x_range, dtype=torch.float32)
        x_tensor = torch.unsqueeze(x_tensor, 0)  # Add a new dimension at index 0
        x_tensor = torch.unsqueeze(x_tensor, 0)  # Add another new dimension at index 0
        x_tensor = torch.unsqueeze(x_tensor, 0)  # Add another new dimension at index 0

        y = activation_fn(torch.tensor(x_tensor, dtype=torch.float32).to("cuda"))
        y = y.squeeze()
        y = y.to("cpu")
        y = y.detach().numpy()

        z_array = np.full_like(x_range, epoch)

        alpha = 0.2 + 0.6 * (epoch / (PlotLen - 1))

        ax.plot(z_array, x_range, y, label=str(activation_fn), alpha=alpha)

    ax.plot(x_range, np.zeros_like(x_range), [0] * len(x_range), color='black', linestyle='dashed', linewidth=1,
            alpha=0.5)

    ax.set_xlim(0, PlotLen - 1)
    ax.set_ylim(-6, 6)
    ax.set_zlim(-6, 6)

    ax.set_xlabel(z)
    ax.set_ylabel('Input (X)')
    ax.set_zlabel('Activation Output (Y)')

    ax.set_xticks(np.arange(PlotLen))
    ax.set_xticklabels(num_epoch_list)

    # ax.legend(loc='upper right', fontsize='small')
    ax.grid(True, linestyle='--', alpha=0.7)

    xx, zz = np.meshgrid(x_range, np.arange(PlotLen))
    yy = np.zeros_like(xx)
    ax.plot_surface(zz, xx, yy, color='gray', alpha=0.2)

    ax.xaxis.set_pane_color((0.9, 0.9, 0.9, 1.0))
    ax.yaxis.set_pane_color((0.9, 0.9, 0.9, 1.0))
    ax.zaxis.set_pane_color((0.9, 0.9, 0.9, 1.0))

    plt.tight_layout()
    plt.show()



