import torch
from torch import nn
import torch.nn.init as init
import matplotlib

matplotlib.use("Agg")

""" 
RepAct Origin
"""
class RepAct_Origin(nn.Module):
    def __init__(self):
        super(RepAct_Origin, self).__init__()
        self.ActorNum = 4
        self.ActorAtn = nn.Parameter(torch.zeros(self.ActorNum) + 1 / (self.ActorNum))
        self.actor_Identity = nn.Identity()
        self.actor_ReLU = nn.ReLU()
        self.actor_PReLU = nn.PReLU()
        self.actor_Hardswish = nn.Hardswish()

    def forward(self, x, plot=False):
        x_Identity = self.actor_Identity(x.clone()) * self.ActorAtn[0]
        x_ReLU = self.actor_ReLU(x.clone()) * self.ActorAtn[1]
        x_Hardswish = self.actor_Hardswish(x.clone()) * self.ActorAtn[2]
        x_PReLU = self.actor_PReLU(x.clone()) * self.ActorAtn[3]
        x = x_ReLU + x_Identity + x_Hardswish + x_PReLU
        return x

    def plotRepAct(self, strTitle, strSave):
        import numpy as np
        # Generate input values from -5 to 5
        x = np.linspace(-7, 7, 100)
        # Create a new figure
        import matplotlib.pyplot as plt
        plt.figure()
        y = self.forward(torch.tensor(x, dtype=torch.float32).to("cuda"), plot=True)
        y = y.to("cpu")
        y = y.detach().numpy()
        # Plot the ReLU activation function
        plt.plot(x, y)  # Reduce alpha to make lines more transparent
        # Set labels and title for the plot
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('RepActivation T1 with ' + strTitle)
        plt.axhline(0, color='gray', linewidth=0.01, linestyle='--')
        plt.axvline(0, color='gray', linewidth=0.01, linestyle='--')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(strSave)
        plt.close()

""" 
RepAct Softmax 
"""
class RepAct_Softmax(nn.Module):
    def __init__(self):
        super(RepAct_Softmax, self).__init__()
        self.ActorNum = 4
        self.ActorAtn = nn.Parameter(torch.zeros(self.ActorNum) + 1 / (self.ActorNum))
        self.actor_Identity = nn.Identity()
        self.actor_ReLU = nn.ReLU()
        self.actor_PReLU = nn.PReLU()
        self.actor_Hardswish = nn.Hardswish()

    def forward(self, x, plot=False):
        weights = torch.softmax(self.ActorAtn, dim=0)
        x_Identity = self.actor_Identity(x.clone()) * weights[0]
        x_ReLU = self.actor_ReLU(x.clone()) * weights[1]
        x_PReLU = self.actor_PReLU(x.clone()) * weights[2]
        x_Hardswish = self.actor_Hardswish(x.clone()) * weights[3]
        x = x_ReLU + x_Identity + x_Hardswish + x_PReLU
        return x

    def plotRepAct(self, strTitle, strSave):
        import numpy as np
        # Generate input values from -5 to 5
        x = np.linspace(-7, 7, 100)
        # Create a new figure
        import matplotlib.pyplot as plt
        plt.figure()
        y = self.forward(torch.tensor(x, dtype=torch.float32).to("cuda"), plot=True)
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

""" 
RepAct BN
"""
class RepAct_BN(nn.Module):
    def __init__(self):
        super(RepAct_BN, self).__init__()
        self.ActorNum = 4
        self.ActorAtn = nn.Parameter(torch.zeros(self.ActorNum) + 1 / (self.ActorNum))
        self.actor_Identity = nn.Identity()
        self.actor_ReLU = nn.ReLU()
        self.actor_PReLU = nn.PReLU()
        self.actor_Hardswish = nn.Hardswish()
        self.bn = nn.BatchNorm2d(num_features=1)

    def forward(self, x, plot=False):
        x_Identity = self.actor_Identity(x.clone()) * self.ActorAtn[0]
        x_ReLU = self.actor_ReLU(x.clone()) * self.ActorAtn[1]
        x_Hardswish = self.actor_Hardswish(x.clone()) * self.ActorAtn[2]
        x_PReLU = self.actor_PReLU(x.clone()) * self.ActorAtn[3]
        x = x_ReLU + x_Identity + x_Hardswish + x_PReLU
        if (len(x.size()) == 4):  # cnn
            bs, ch, height, width = x.size()
            x_stacked = x.view(bs, 1, 1, ch * height * width)
            x_stacked = self.bn(x_stacked)
            x_stacked = x_stacked.view(bs, ch, height, width)
        if (len(x.size()) == 3):  # transformer
            bs, s1, s2 = x.size()
            x_stacked = x.view(bs, 1, 1, s1 * s2)
            x_stacked = self.bn(x_stacked)
            x_stacked = x_stacked.view(bs, s1, s2)
        return x_stacked

    def plotRepAct(self, strTitle, strSave):
        import numpy as np
        # Generate input values from -5 to 5
        x = np.linspace(-7, 7, 100)
        # Create a new figure
        import matplotlib.pyplot as plt
        plt.figure()
        # Reshape x to (1, 1, 1, 100) using torch.unsqueeze
        x_tensor = torch.tensor(x, dtype=torch.float32)
        x_tensor = torch.unsqueeze(x_tensor, 0)  # Add a new dimension at index 0
        x_tensor = torch.unsqueeze(x_tensor, 0)  # Add another new dimension at index 0
        x_tensor = torch.unsqueeze(x_tensor, 0)  # Add another new dimension at index 0
        y = self.forward(torch.tensor(x_tensor, dtype=torch.float32).to("cuda"), plot=True)
        y = y.squeeze()
        y = y.to("cpu")
        y = y.detach().numpy()
        # Plot the ReLU activation function
        plt.plot(x, y)  # Reduce alpha to make lines more transparent
        # Set labels and title for the plot
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('RepAct_BN_AfterAdd with ' + strTitle)
        plt.axhline(0, color='gray', linewidth=0.01, linestyle='--')
        plt.axvline(0, color='gray', linewidth=0.01, linestyle='--')
        plt.grid(True, linestyle='--', alpha=0.7)

        plt.savefig(strSave)
        plt.close()
