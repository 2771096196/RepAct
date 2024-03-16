import os
import torch


def extract_layer_number(file_name):
    parts = file_name.split('.')
    parts = parts[0].split('_')[-1]
    return int(parts)


def list_subfolders(folder_path):
    subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
    return subfolders


def load_and_sort_files(folder_path):
    file_dict_list = []

    sorted_files = sorted(os.listdir(folder_path), key=extract_layer_number)

    for file_name in sorted_files:
        if file_name.endswith('.pth'):
            file_path = os.path.join(folder_path, file_name)
            try:
                checkpoint = torch.load(file_path)
                file_dict_list.append({file_name: checkpoint})
                # print(f"Loaded weights from {file_name}")
            except Exception as e:
                print(f"Error loading weights from {file_name}: {e}")

    return file_dict_list


def get_epoch_pths_list(base_folder_path):
    epochList = list_subfolders(base_folder_path)
    EpochPthes_List = []

    for i in epochList:
        folder_path = os.path.join(base_folder_path, i)
        loaded_files = load_and_sort_files(folder_path)
        EpochPthes_List.append({str(i): loaded_files})

    return EpochPthes_List


def getValue(dict, index):
    return list(dict[index].values())[0]


def partition_and_get_values_auto(input_list, Nums):
    total_length = len(input_list)
    partition_size = int((total_length - 2) / (Nums - 2 + 1) + 0.5)
    if (Nums < 2):
        print("Nums<2")
        return None
    if (Nums > total_length):
        print("Nums>total_length")
        return None
    target_positions = [0]
    temp = 0
    for i in range(Nums - 2):
        temp = temp + partition_size
        target_positions.append(temp)
    target_positions.append(total_length - 1)

    values = []
    for position in target_positions:
        values.append(input_list[position])

    return values


def getIndexEpochRepAct(list, index):
    list = sorted(list, key=lambda x: int(list(x.keys())[0]))
    return getValue(list, index)


def getEachEpochIndexRepAct(list_in, index):
    list_res = []
    list_in = sorted(list_in, key=lambda x: int(list(x.keys())[0]))
    for i in range(len(list_in)):
        list_res.append({i: getValue(list(list_in[i].values())[0], index)})
        print((list(list_in[i].values())[0][index].keys()))
    return list_res
