import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torch import nn, Tensor

import torch.nn.init as init


class RepAct_Origin(nn.Module):
    def __init__(self, initWeight):
        super(RepAct_Origin, self).__init__()
        self.ActorNum = 4
        initWeight = torch.tensor(initWeight)
        self.ActorAtn = nn.Parameter(initWeight)
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


class RepAct_BN(nn.Module):
    def __init__(self, initWeight):
        super(RepAct_BN, self).__init__()
        self.ActorNum = 4
        initWeight = torch.tensor(initWeight)
        self.ActorAtn = nn.Parameter(initWeight)
        # self.ActorAtn = nn.Parameter(torch.zeros(self.ActorNum) + 1 / (self.ActorNum))
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
        x_stacked = self.bn(x.view(1, 1, 1, -1))
        return x_stacked.view(-1)


class RepAct_Softmax(nn.Module):
    def __init__(self, initWeight):
        super(RepAct_Softmax, self).__init__()
        self.ActorNum = 4
        self.ActorAtn = nn.Parameter(torch.tensor(initWeight))
        # self.ActorAtn = nn.Parameter(torch.zeros(self.ActorNum) + 1 / (self.ActorNum))
        self.actor_Identity = nn.Identity()
        self.actor_ReLU = nn.ReLU()
        self.actor_PReLU = nn.PReLU()
        self.actor_Hardswish = nn.Hardswish()

    def forward(self, x):
        weights = torch.softmax(self.ActorAtn, dim=0)
        x_Identity = self.actor_Identity(x.clone()) * weights[0]
        x_ReLU = self.actor_ReLU(x.clone()) * weights[1]
        x_PReLU = self.actor_PReLU(x.clone()) * weights[2]
        x_Hardswish = self.actor_Hardswish(x.clone()) * weights[3]
        x = x_ReLU + x_Identity + x_Hardswish + x_PReLU
        return x


name = 'RepAct-Ⅰ'
# name = 'RepAct-Ⅱ'
# name = 'RepAct-Ⅲ'
# Generate input values from -5 to 5
x = torch.linspace(-5, 5, 100)
x.requires_grad_(True)

# Create a new figure
plt.figure(figsize=(6, 5))

len = 5
for i in range(len):
    for j in range(len):
        for k in range(len):
            for f in range(len):
                # Plot the RepActivation with 200-times Random Init
                if (name == 'RepAct-Ⅰ'):
                    my_actor = RepAct_Origin(initWeight=[i / (len - 1), j / (len - 1), k / (len - 1), f / (len - 1)])
                if (name == 'RepAct-Ⅱ'):
                    my_actor = RepAct_Softmax(initWeight=[i / (len - 1), j / (len - 1), k / (len - 1), f / (len - 1)])
                if (name == 'RepAct-Ⅲ'):
                    my_actor = RepAct_BN(initWeight=[i / (len - 1), j / (len - 1), k / (len - 1), f / (len - 1)])
                y = my_actor(x)
                y = y.detach().numpy()
                plt.plot(x.detach().numpy(), y, alpha=0.2)

# Set labels and title for the plot
plt.xlabel('x')
plt.title(name)
plt.axhline(0, color='gray', linewidth=0.00001, linestyle='-.')
plt.axvline(0, color='gray', linewidth=0.00001, linestyle='-.')
plt.grid(True)

# Add legends
plt.legend([name], loc='upper left')

# Adjust layout
plt.tight_layout()

# Show the combined plot
plt.savefig(name + '.png')

# Create a new figure
plt.figure(figsize=(6, 5))
for i in range(len):
    for j in range(len):
        for k in range(len):
            for f in range(len):
                # Plot the RepActivation with 200-times Random Init
                my_actor = RepAct_Origin(initWeight=[i / (len - 1), j / (len - 1), k / (len - 1), f / (len - 1)])

                # Plot the Derivative of RepActivation with 200-times Random Init
                y = my_actor(x)
                y.sum().backward()
                grad_x = x.grad.numpy()
                plt.plot(x.detach().numpy(), grad_x, alpha=0.2)
                # Clear the gradients for the next iteration
                my_actor.zero_grad()
                x.grad.zero_()

# Set labels and title for the plot
plt.xlabel('x')
plt.title('Derivative')
plt.axhline(0, color='gray', linewidth=0.00001, linestyle='-.')
plt.axvline(0, color='gray', linewidth=0.00001, linestyle='-.')
plt.grid(True)

# Add legends
plt.legend(['Derivative'], loc='upper left')
# Adjust layout
plt.tight_layout()

# Show the combined plot
plt.savefig(name + '-Derivative.png')
