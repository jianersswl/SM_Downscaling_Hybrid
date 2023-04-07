# I/O
import os
import re
import numpy as np
# Pytorch
from torch.utils.data import Dataset

class SMAPDataset(Dataset):
    '''
    root: 输入数据的根目录
    sequence: 存储有效天数与其对应SMAPID的字典
    '''
    def __init__(self, root, sequence):
        # 输入变量
        self.smap = []       # SMAP数据路径
        self.texture = []     # 土壤质地数据路径
        
        # label变量
        self.sm = []        # 地面真实 SM 数据路径
        self.smap_unorm = []  # SMAP数据的非归一化值路径
        self.ati = []       # 包含每个站点中的 [ati, atim, atisd]
        
        # 元信息
        self.full_day_sequence = sequence.keys()  # 有效天数的完整序列
        self.full_insitu_sequence = set()       # 有效的 in-situ 序列
        self.full_smapid_sequence = set()       # 有效的 SMAPID 序列
        
        print('***************************Load data path******************************')
        for i in sequence.keys(): # 例如：2015187
            print('_______________________________' + str(i) + '_______________________________')
            for j in sequence[i]: # 例如：SMAPID=1
                print('_____________________________smap cell: ' + str(j) + '_____________________________')
                # 将出现的SMAPID加入full_smapid_sequence
                self.full_smapid_sequence.add(j)
                
                # 加输入变量路径
                self.smap.append(root + 'INPUT\\SMAP\\' + i + '\\' + str(j) + '.npy')
                self.texture.append(root + 'INPUT\\TEXTURE\\' + str(j) + '.npy')
                
                # 显示添加的路径
                print_path(root + 'INPUT\\SMAP\\' + i + '\\' + str(j) + '.npy')
                print_path(root + 'INPUT\\TEXTURE\\' + str(j) + '.npy')
                      
                # 一个 SMAP 对应多个 in-situ SM
                self.smap_unorm.append(root + 'LABEL\\SMAP\\' + i + '\\' + str(j) + '.npy')
                smap_to_insitu = np.load(root + "LABEL\\SMAPID2INSITUID\\" + i + '\\' + str(j) + '.npy')
                insitu_sm_list = []
                insitu_ati_list = []
                for _id in smap_to_insitu:
                    # 将出现的in-situ id加入full_smapid_sequence
                    self.full_insitu_sequence.add(_id)
                    
                    # 将路径加入列表
                    insitu_sm_list.append(root + "LABEL\\SM\\" + i + "\\" + str(_id) + ".npy")
                    insitu_ati_list.append(root + "LABEL\\ATI\\" + i + "\\" + str(_id) + ".npy")
                    
                    # 显示添加的路径
                    print_path(root + "LABEL\\SM\\" + i + "\\" + str(_id) + ".npy")
                    print_path(root + "LABEL\\ATI\\" + i + "\\" + str(_id) + ".npy")
                      
                # add the data of insitu in insitu_list
                self.sm.append(insitu_sm_list)
                self.ati.append(insitu_ati_list)    
                      
    def __getitem__(self, idx):
        # 定义数据package
        data_pkg = {'processed_data': [], 'label_data': [], 'meta_data': {}}
        
        # 获取input数据路径
        smap_path = self.smap[idx]
        texture_path = self.texture[idx]
        
        # 通过路径获取日期和SMAPID
        date = re.findall(r'\d+', smap_path)[-2] # //2015091//0.npy
        smapid = os.path.basename(smap_path).split('.')[0]
        data_pkg['meta_data']['date'] = date
        data_pkg['meta_data']['smapid'] = smapid
        
        # 加载input数据
        smap = np.load(smap_path)
        texture = np.load(texture_path)
        
        # 选择展平为连接input特征的方式
        x = self.__stack__(smap, texture)
        data_pkg['processed_data'] = x
        
        # 加载label数据
        smap_unorm_path = self.smap_unorm[idx]
        smap_unorm = np.load(smap_unorm_path)
        
        # 一个SMAPID可能对应多个in-situ sm
        sm_list = self.sm[idx]
        ati_list = self.ati[idx]
        data_pkg['meta_data']['insituid'] = []
        for i in range(len(sm_list)):
            # 获取数据路径
            sm_path = sm_list[i]
            ati_path = ati_list[i]
            # 通过路径获取站点id
            insituid = re.findall(r'\d+', sm_path)[-1] #//2015091//1.npy
            data_pkg['meta_data']['insituid'].append(insituid)
            # 加载数据
            sm = np.load(sm_path, allow_pickle=True)
            ati = np.load(ati_path, allow_pickle=True)
            # 将一个站点所需的label数据list加入package
            data_pkg['label_data'].append([sm, smap_unorm, ati])      # other_data -> [[sm, smap, ati], ...], 
                                                                    # sm -> [float]
                                                                    # smap -> [float], 
                                                                    # ati -> [ati, atim, atisd]
            
        return data_pkg

    def __len__(self):
        return len(self.smap)

    ### the way to concatenate input data
    def __stack__(self, smap:np.array, texture:np.array)->np.array:
        shape = [texture.shape[0],texture.shape[0], 1]
        extend_smap = np.ones(shape)*smap[0]
        conca_data = np.concatenate((extend_smap, texture[:, :, 1:-1]), axis=2)
        return conca_data
    
    def __flatten__(self, smap:np.array, texture:np.array)->np.array:
        # normalization is done before loading
        texture_flat = texture.flatten()
        return  np.concatenate((smap, texture_flat), axis=0)
    
    def get_input_shape(self, idx):
        data_pkg = self.__getitem__(idx)
        return data_pkg['processed_data'].shape
    
    def get_full_insitu_sequence(self):
        return self.full_insitu_sequence
    
    def get_full_smapid_sequence(self):
        return self.full_smapid_sequence
    
    def get_full_day_sequence(self):
        return self.full_day_sequence
    
def print_path(path):
    if os.path.exists(path):
        print("\033[32m" + path + "\033[0m")  # 绿色文本
    else:
        print("\033[31m" + path + "\033[0m")  # 红色文本 