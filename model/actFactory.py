from RepActs.RepActs import *
from RepActs.validRep.RepAct_Softmax_Reconstruct import *
from RepActs.validRep.RepAct_Origin_Reconstruct import *
from RepActs.validRep.RepAct_BN_Reconstruct import *
from model.activations import *

"""
Return the corresponding activation function instance 
based on the input activation function name, 
create the corresponding activation function 
instance through the factory class, and return
"""


class ActivationFactory:
    @staticmethod
    def create_activation(act, inChannel):
        if act == "RepAct_Softmax":
            return RepAct_Softmax()
        elif act == "RepAct_Origin":
            return RepAct_Origin()
        elif act == "RepAct_BN":
            return RepAct_BN()
        # valid ReParam
        elif act == "RepAct_Softmax_Reconstruct":
            return RepAct_Softmax_Reconstruct()
        elif act == "RepAct_Origin_Reconstruct":
            return RepAct_Origin_Reconstruct()
        elif act == "RepAct_BN_Reconstruct":
            return RepAct_BN_Reconstruct()

        elif act == "Identity":
            return nn.Identity()
        elif act == "ReLU":
            return nn.ReLU()
        elif act == "HardSwish":
            return nn.Hardswish()
        elif act == "LReLU":
            return nn.LeakyReLU()
        elif act == "PReLU":
            return nn.PReLU()
        elif act == "FReLU":
            return FReLU(in_channels=inChannel)
        elif act == "DYReLU":
            return DyReLUA(channels=inChannel, reduction=4, k=2, conv_type='2d')
        elif act == "AReLU":
            return AReLU()
        elif act == "SiLU":
            return nn.SiLU()
        elif act == "Softplus":
            return nn.Softplus()
        elif act == "Swish":
            return Swish()
        elif act == "Mish":
            return nn.Mish()
        elif act == "ELU":
            return nn.ELU()
        elif act == "SELU":
            return nn.SELU()
        elif act == "GELU":
            return nn.GELU()
        elif act == "CELU":
            return nn.CELU()
        elif act == "AconC":
            return AconC(width=1)
        else:
            print('error in act')
            return None
