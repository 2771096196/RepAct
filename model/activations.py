import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.parameter import Parameter
import torch.nn.functional as F


class FReLU(nn.Module):
    r""" FReLU formulation. The funnel condition has a window size of kxk. (k=3 by default)
    """

    def __init__(self, in_channels):
        super().__init__()
        self.conv_frelu = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels)
        self.bn_frelu = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        y = self.conv_frelu(x)
        y = self.bn_frelu(y)
        x = torch.max(x, y)
        return x


class AReLU(nn.Module):
    def __init__(self, alpha=0.90, beta=2.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor([alpha]))
        self.beta = nn.Parameter(torch.tensor([beta]))

    def forward(self, input):
        alpha = torch.clamp(self.alpha, min=0.01, max=0.99)
        beta = 1 + torch.sigmoid(self.beta)

        return F.relu(input) * beta - F.relu(-input) * alpha


class DyReLU(nn.Module):
    def __init__(self, channels, reduction=4, k=2, conv_type='2d'):
        super(DyReLU, self).__init__()
        self.channels = channels
        self.k = k
        self.conv_type = conv_type
        assert self.conv_type in ['1d', '2d']

        self.fc1 = nn.Linear(channels, channels // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels // reduction, 2 * k)
        self.sigmoid = nn.Sigmoid()

        self.register_buffer('lambdas', torch.Tensor([1.] * k + [0.5] * k).float())
        self.register_buffer('init_v', torch.Tensor([1.] + [0.] * (2 * k - 1)).float())

    def get_relu_coefs(self, x):
        theta = torch.mean(x, axis=-1)
        if self.conv_type == '2d':
            theta = torch.mean(theta, axis=-1)
        theta = self.fc1(theta)
        theta = self.relu(theta)
        theta = self.fc2(theta)
        theta = 2 * self.sigmoid(theta) - 1
        return theta

    def forward(self, x):
        raise NotImplementedError


class DyReLUA(DyReLU):
    def __init__(self, channels, reduction=4, k=2, conv_type='2d'):
        super(DyReLUA, self).__init__(channels, reduction, k, conv_type)
        self.fc2 = nn.Linear(channels // reduction, 2 * k)

    def forward(self, x):
        assert x.shape[1] == self.channels
        theta = self.get_relu_coefs(x)

        relu_coefs = theta.view(-1, 2 * self.k) * self.lambdas + self.init_v
        # BxCxL -> LxCxBx1
        x_perm = x.transpose(0, -1).unsqueeze(-1)
        output = x_perm * relu_coefs[:, :self.k] + relu_coefs[:, self.k:]
        # LxCxBx2 -> BxCxL
        result = torch.max(output, dim=-1)[0].transpose(0, -1)

        return result


class Swish(torch.nn.Module):
    __constants__ = ['num_parameters']
    num_parameters: int

    def __init__(self, num_parameters: int = 1, init: float = 0.25) -> None:
        self.num_parameters = num_parameters
        super(Swish, self).__init__()
        self.weight = Parameter(torch.Tensor(num_parameters).fill_(init))

    def forward(self, input: Tensor) -> Tensor:
        return input * F.sigmoid(self.weight * input)

    def extra_repr(self) -> str:
        return 'num_parameters={}'.format(self.num_parameters)

    # def forward(self, input: Tensor) -> Tensor:
    #    return input * torch.sigmoid(input)


class Elliott(torch.nn.Module):
    __constants__ = ['inplace']
    inplace: bool

    def __init__(self, inplace: bool = False):
        super(Elliott, self).__init__()
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        return 0.5 * input / (1 + torch.abs(input)) + 0.5

    def extra_repr(self) -> str:
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str


class ABReLU(torch.nn.Module):
    __constants__ = ['inplace']
    inplace: bool

    def __init__(self, inplace: bool = False):
        super(ABReLU, self).__init__()
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        # print(input.shape)
        input1 = torch.mean(input, dim=(0, 1, 2, 3))
        # input1 = torch.mean(input.view(input.size(0), -1), dim=2)
        # input1 = F.adaptive_avg_pool2d(input, (1, 1, 1))
        # print(input1.shape)
        # print(input1)
        # print(input1(1,1,1,1:10))
        input = input - input1
        return F.relu(input, inplace=self.inplace)

    def extra_repr(self) -> str:
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str


class LiSHT(torch.nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return input * torch.tanh(input)


class Mish(torch.nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return input * F.tanh(F.softplus(input))


class SRS(torch.nn.Module):
    __constants__ = ['num_parameters']
    num_parameters: int

    #    def __init__(self, num_parameters: int = 1, init: float = (5.0,3.0)) -> None:
    def __init__(self, num_parameters: int = 1, init: float = (10.0, 10.0)) -> None:  # for SENet18
        self.num_parameters = num_parameters
        super(SRS, self).__init__()
        self.weight1 = Parameter(torch.Tensor(num_parameters).fill_(init[0]))
        self.weight2 = Parameter(torch.Tensor(num_parameters).fill_(init[1]))

    def forward(self, input: Tensor) -> Tensor:
        # self.weight1 = torch.abs(self.weight1)
        # self.weight2 = torch.abs(self.weight2)
        # w1 = self.weight1 + 1e-8
        # print(w1.shape)
        # print(input.shape)
        # a = torch.div(input,abs(self.weight1+1e-8))
        # b = torch.exp(- torch.div(input,abs(self.weight2+1e-8)))
        # return torch.div(input, a + b + 1e-8)
        return torch.div(input, 1e-2 + torch.div(input, torch.abs(self.weight1) + 1e-2) + torch.exp(
            -torch.div(input, torch.abs(self.weight2) + 1e-2)))

    def extra_repr(self) -> str:
        return 'num_parameters={}'.format(self.num_parameters)


class PDELU(torch.nn.Module):
    __constants__ = ['num_parameters']
    num_parameters: int

    def __init__(self, num_parameters: int = 1, init: float = 1.0) -> None:
        self.num_parameters = num_parameters
        super(PDELU, self).__init__()
        self.weight = Parameter(torch.Tensor(num_parameters).fill_(init))

    def forward(self, input: Tensor) -> Tensor:
        input1 = input
        input1[input < 0] = 0
        input2 = self.weight * (torch.pow(1 + 0.1 * input, 10) - 1)
        input2[input >= 0] = 0
        return input1 + input2

    def extra_repr(self) -> str:
        return 'num_parameters={}'.format(self.num_parameters)


class AconC(nn.Module):
    r""" ACON activation (activate or not).
    # AconC: (p1*x-p2*x) * sigmoid(beta*(p1*x-p2*x)) + p2*x, beta is a learnable parameter
    # according to "Activate or Not: Learning Customized Activation" <https://arxiv.org/pdf/2009.04759.pdf>.
    """

    def __init__(self, width=1):
        super().__init__()
        self.p1 = nn.Parameter(torch.randn(1, width, 1, 1))
        self.p2 = nn.Parameter(torch.randn(1, width, 1, 1))
        self.beta = nn.Parameter(torch.ones(1, width, 1, 1))

    def forward(self, x):
        return (self.p1 * x - self.p2 * x) * torch.sigmoid(self.beta * (self.p1 * x - self.p2 * x)) + self.p2 * x


class MetaAconC(nn.Module):
    r""" ACON activation (activate or not).
    # MetaAconC: (p1*x-p2*x) * sigmoid(beta*(p1*x-p2*x)) + p2*x, beta is generated by a small network
    # according to "Activate or Not: Learning Customized Activation" <https://arxiv.org/pdf/2009.04759.pdf>.
    """

    def __init__(self, width, r=1):
        super().__init__()
        self.fc1 = nn.Conv2d(width, max(r, width // r), kernel_size=1, stride=1, bias=True)
        self.bn1 = nn.BatchNorm2d(max(r, width // r))
        self.fc2 = nn.Conv2d(max(r, width // r), width, kernel_size=1, stride=1, bias=True)
        self.bn2 = nn.BatchNorm2d(width)

        self.p1 = nn.Parameter(torch.randn(1, width, 1, 1))
        self.p2 = nn.Parameter(torch.randn(1, width, 1, 1))

    def forward(self, x):
        beta = torch.sigmoid(
            self.bn2(self.fc2(self.bn1(self.fc1(x.mean(dim=2, keepdims=True).mean(dim=3, keepdims=True))))))
        return (self.p1 * x - self.p2 * x) * torch.sigmoid(beta * (self.p1 * x - self.p2 * x)) + self.p2 * x
