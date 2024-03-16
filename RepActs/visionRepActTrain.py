import os
import shutil

from RepActs.RepActs import *
from RepActs.validRep.RepAct_BN_Reconstruct import RepAct_BN_Reconstruct
from RepActs.validRep.RepAct_Origin_Reconstruct import RepAct_Origin_Reconstruct
from RepActs.validRep.RepAct_Softmax_Reconstruct import RepAct_Softmax_Reconstruct


def plotRepAct_all(model, strTitle, strSave):
    import numpy as np
    # Generate input values from -5 to 5
    x = np.linspace(-7, 7, 100)
    # Create a new figure
    import matplotlib.pyplot as plt
    plt.figure()
    x_tensor = torch.tensor(x, dtype=torch.float32)
    x_tensor = torch.unsqueeze(x_tensor, 0)  # Add a new dimension at index 0
    x_tensor = torch.unsqueeze(x_tensor, 0)  # Add another new dimension at index 0
    x_tensor = torch.unsqueeze(x_tensor, 0)  # Add another new dimension at index 0
    y = model(torch.tensor(x_tensor, dtype=torch.float32).to("cuda"))
    y = y.squeeze()
    y = y.to("cpu")
    y = y.detach().numpy()
    # Plot the ReLU activation function
    plt.plot(x, y)  # Reduce alpha to make lines more transparent
    # Set labels and title for the plot
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('RepActivation Softmax with ' + strTitle)
    plt.axhline(0, color='gray', linewidth=0.01, linestyle='--')
    plt.axvline(0, color='gray', linewidth=0.01, linestyle='--')
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.savefig(strSave)
    plt.close()

def plot_repact_layers(model, folder_name, epoch):
    """
    Plot the RepAct for each layer in the model.
    Parameters:
        model (torch.nn.Module): The PyTorch model containing RepAct_Origin layers.
        epoch (int): The epoch number used for creating the folder and filenames.
    """
    model.eval()
    model_list = list(model.modules())

    # Check if the folder exists
    if os.path.exists(folder_name):
        # If it exists, remove the folder and its contents
        shutil.rmtree(folder_name)
        print(f"Folder '{folder_name}' removed successfully.")

    # Create the folder
    os.makedirs(folder_name)
    print(f"Folder '{folder_name}' created successfully.")

    flag = 0
    for module in model_list:
        flag = flag + 1
        if isinstance(module, RepAct_Origin):
            module.plotRepAct("epoch:" + str(epoch) + "___layer:" + str(flag),
                              folder_name + "/" + "epoch_" + str(epoch) + "_layer_" + str(flag) + ".jpg")
            torch.save(module, folder_name + "/" + "epoch_" + str(epoch) + "_layer_" + str(flag) + ".pth")

        if isinstance(module, RepAct_Softmax):
            module.plotRepAct("epoch:" + str(epoch) + "___layer:" + str(flag),
                              folder_name + "/" + "epoch_" + str(epoch) + "_layer_" + str(flag) + ".jpg")
            torch.save(module, folder_name + "/" + "epoch_" + str(epoch) + "_layer_" + str(flag) + ".pth")

        if isinstance(module, RepAct_BN):
            module.plotRepAct("epoch:" + str(epoch) + "___layer:" + str(flag),
                              folder_name + "/" + "epoch_" + str(epoch) + "_layer_" + str(flag) + ".jpg")

            torch.save(module, folder_name + "/" + "epoch_" + str(epoch) + "_layer_" + str(flag) + ".pth")

        if isinstance(module, RepAct_Softmax_Reconstruct):
            module.plotRepAct("epoch:" + str(epoch) + "___layer:" + str(flag),
                              folder_name + "/" + "epoch_" + str(epoch) + "_layer_" + str(flag) + ".jpg")

            torch.save(module, folder_name + "/" + "epoch_" + str(epoch) + "_layer_" + str(flag) + ".pth")

        if isinstance(module, RepAct_Origin_Reconstruct):
            module.plotRepAct("epoch:" + str(epoch) + "___layer:" + str(flag),
                              folder_name + "/" + "epoch_" + str(epoch) + "_layer_" + str(flag) + ".jpg")

            torch.save(module, folder_name + "/" + "epoch_" + str(epoch) + "_layer_" + str(flag) + ".pth")

        if isinstance(module, RepAct_BN_Reconstruct):
            module.plotRepAct("epoch:" + str(epoch) + "___layer:" + str(flag),
                              folder_name + "/" + "epoch_" + str(epoch) + "_layer_" + str(flag) + ".jpg")

            torch.save(module, folder_name + "/" + "epoch_" + str(epoch) + "_layer_" + str(flag) + ".pth")
