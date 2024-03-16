import torch
import torch.nn as nn

class RepAct_BN_Reconstruct(nn.Module):
    def __init__(self, inference=False):
        super(RepAct_BN_Reconstruct, self).__init__()
        self.inference = inference
        self.ActorNum = 4
        self.ActorAtn = nn.Parameter(torch.zeros(self.ActorNum) + 1 / (self.ActorNum)) * 5
        # self.ActorAtn = nn.Parameter(torch.rand(self.ActorNum))
        self.actor_Identity = nn.Identity()
        self.actor_ReLU = nn.ReLU()
        self.actor_PReLU = nn.PReLU()
        self.actor_Hardswish = nn.Hardswish()
        self.bn = nn.BatchNorm2d(num_features=1)
        self.randomBNInit()
        self.x_0_list = []
        self.x_1_list = []
        self.x_2_list = []

    def randomBNInit(self):
        self.bn.running_mean = torch.tensor([0.7254, ])
        self.bn.running_var = torch.tensor([1.7242, ])
        self.bn.bias = nn.Parameter(torch.tensor(1.2524))
        self.bn.weight = nn.Parameter(torch.tensor(1.3254))

    def RepActFuse(self):
        # ********************BN*********************
        mean = self.bn.running_mean
        var_sqrt = torch.sqrt(self.bn.running_var + self.bn.eps)
        gamma = self.bn.weight
        beta = self.bn.bias
        gamma_by_var_sqrt = gamma / var_sqrt
        weights = self.ActorAtn
        # X ^ 0 coefficient
        x_0_GT_3 = nn.Parameter(gamma_by_var_sqrt * torch.tensor(0.) - mean * gamma_by_var_sqrt + beta)
        x_0_GT_0_LT_3 = nn.Parameter(gamma_by_var_sqrt * torch.tensor(0.) - mean * gamma_by_var_sqrt + beta)
        x_0_GT_N3_LT_0 = nn.Parameter(gamma_by_var_sqrt * torch.tensor(0.) - mean * gamma_by_var_sqrt + beta)
        x_0_LT_N3 = nn.Parameter(gamma_by_var_sqrt * torch.tensor(0.) - mean * gamma_by_var_sqrt + beta)
        self.x_0_list = [x_0_GT_3, x_0_GT_0_LT_3, x_0_GT_N3_LT_0, x_0_LT_N3]
        # X ^ 1 coefficient
        x_1_GT_3 = gamma_by_var_sqrt * nn.Parameter(weights[0] + weights[1] + weights[2] + weights[3])
        x_1_GT_0_LT_3 = gamma_by_var_sqrt * nn.Parameter(weights[0] + weights[1] + weights[2] + 0.5 * weights[3])
        x_1_GT_N3_LT_0 = gamma_by_var_sqrt * nn.Parameter(
            weights[0] + 0 * weights[1] + self.actor_PReLU.weight.data * weights[2] + 0.5 * weights[3])
        x_1_LT_N3 = gamma_by_var_sqrt * nn.Parameter(
            weights[0] + 0 * weights[1] + self.actor_PReLU.weight.data * weights[2] + 0 * weights[3])
        self.x_1_list = [x_1_GT_3, x_1_GT_0_LT_3, x_1_GT_N3_LT_0, x_1_LT_N3]
        # X ^ 2 coefficient
        x_2_GT_3 = gamma_by_var_sqrt * nn.Parameter(torch.tensor(0.))
        x_2_GT_0_LT_3 = gamma_by_var_sqrt * nn.Parameter((1 / 6) * weights[3])
        x_2_GT_N3_LT_0 = gamma_by_var_sqrt * nn.Parameter((1 / 6) * weights[3])
        x_2_LT_N3 = gamma_by_var_sqrt * nn.Parameter(torch.tensor(0.))
        self.x_2_list = [x_2_GT_3, x_2_GT_0_LT_3, x_2_GT_N3_LT_0, x_2_LT_N3]

    def forward(self, x, plot=False):
        if (self.inference):
            x_clone = x.clone()
            x[x_clone >= 3] = self.x_0_list[0] + \
                              x[x_clone >= 3] * self.x_1_list[0] + \
                              torch.pow(x[x_clone >= 3], 2) * self.x_2_list[0]
            idx_range_0_3 = (x_clone >= 0) & (x_clone <= 3)
            x[idx_range_0_3] = self.x_0_list[1] + \
                               x[idx_range_0_3] * self.x_1_list[1] + \
                               torch.pow(x[idx_range_0_3], 2) * self.x_2_list[1]
            idx_range_N3_0 = (x_clone >= -3) & (x_clone <= 0)
            x[idx_range_N3_0] = self.x_0_list[2] + \
                                x[idx_range_N3_0] * self.x_1_list[2] + \
                                torch.pow(x_clone[idx_range_N3_0], 2) * self.x_2_list[2]
            x[x_clone <= -3] = self.x_0_list[3] + x[x_clone <= -3] * self.x_1_list[3] + \
                               torch.pow(x[x_clone <= -3], 2) * self.x_2_list[3]
            return x
        else:
            x_Identity = self.actor_Identity(x.clone()) * self.ActorAtn[0]
            x_ReLU = self.actor_ReLU(x.clone()) * self.ActorAtn[1]
            x_PReLU = self.actor_PReLU(x.clone()) * self.ActorAtn[2]
            x_Hardswish = self.actor_Hardswish(x.clone()) * self.ActorAtn[3]
            x = x_ReLU + x_Identity + x_Hardswish + x_PReLU
            bs, ch, height, width = x.size()
            x_stacked = x.view(bs, 1, 1, ch * height * width)
            xx = x.view(bs, 1, 1, ch * height * width)
            x_stacked = self.bn(x_stacked)
            x_stacked = x_stacked.view(bs, ch, height, width)
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

if __name__ == '__main__':
    RepAct_BN_Add_Stack_init = RepAct_BN_Reconstruct().train()
    # Generate random input data
    batch_size = 10
    input_channels = 200
    input_height = 100
    input_width = 100
    input_data = torch.randn(batch_size, input_channels, input_height, input_width)

    print("-------------------DATA-------------------")
    # print(input_data)
    print("-------------------RepAct_BN_Reconstruct_Reconstruct.py-------------------")
    A = RepAct_BN_Add_Stack_init(input_data)

    RepAct_BN_Add_Stack_init.eval()
    A = RepAct_BN_Add_Stack_init(input_data)
    # print(A)
    print("-------------------"
          "       Trans       "
          "-------------------")

    RepAct_BN_Add_Stack_init.inference = True

    RepAct_BN_Add_Stack_init.RepActFuse()
    B = RepAct_BN_Add_Stack_init(input_data)

    import numpy as np

    unequal_mask = A.detach().numpy() - B.detach().numpy() > 0.00001
    count_of_true = np.count_nonzero(unequal_mask)
    print("count_of_true:", count_of_true)
    if unequal_mask.any():
        # If there are unequal values, raise an exception and show the unequal values
        unequal_values_A = A[torch.tensor(unequal_mask)]
        unequal_values_B = B[torch.tensor(unequal_mask)]
        raise ValueError(f"A and B are not equal up to the first 4 decimals at some positions.\n"
                         f"Unequal values in A: {unequal_values_A}\n"
                         f"Unequal values in B: {unequal_values_B}")
    else:
        print("Equal")
