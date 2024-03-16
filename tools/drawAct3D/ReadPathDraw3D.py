from tools.drawAct3D.ReadPath import *
from tools.drawAct3D.TDAct import plot_activation_functions_custom_z

'''
draw layer where Epoch=index 
'''
def draw_each_epoch_list_indexLayer(folder_path="", Nums=3, index=0):
    epoch_pth_list = get_epoch_pths_list(folder_path)
    Each_epoch_list_indexLayer = getEachEpochIndexRepAct(epoch_pth_list, index)
    Each_epoch_list_indexLayer_autoNum = partition_and_get_values_auto(Each_epoch_list_indexLayer, Nums)
    keys_list = [list(d.keys())[0] for d in Each_epoch_list_indexLayer_autoNum]
    values_list = [list(d.values())[0].to("cuda") for d in Each_epoch_list_indexLayer_autoNum]
    plot_activation_functions_custom_z(values_list, keys_list, z='epochs')


'''
draw Epoch where layer=index
'''
def draw_indexEpoch_list_each_Layer(folder_path="", Nums=3, index=0):
    epoch_pth_list = get_epoch_pths_list(folder_path)
    indexEpoch_list_each_Layer = getIndexEpochRepAct(epoch_pth_list, index)
    indexEpoch_list_each_Layer_autoNum = partition_and_get_values_auto(indexEpoch_list_each_Layer, Nums)
    keys_list = [list(d.keys())[0] for d in indexEpoch_list_each_Layer_autoNum]
    values_list = [list(d.values())[0].to("cuda") for d in indexEpoch_list_each_Layer_autoNum]
    plot_activation_functions_custom_z(values_list, keys_list, z='layers')



folder_path = r"./runs/mobilenet_v3_small_RepAct_Origin_Reconstruct____self_train_val__0.0004_0.01_32"
# draw Epoch where layer=index
draw_indexEpoch_list_each_Layer(folder_path=folder_path, Nums=5, index=0)
# draw layer where Epoch=index
draw_each_epoch_list_indexLayer(folder_path=folder_path, Nums=5, index=-1)

