# I/O
import os
import re
import glob
import numpy as np
# Pytorch
from torch.utils.data import Dataset

class SMAPDataset(Dataset):
    '''
    root: 输入数据的根目录
    insitu_validation: 是否用于验证站点数据
    '''
    def __init__(self, root, insitu_validation=False):
        # 数据加载模式
        self.insitu_validation = insitu_validation
        
        # 输入变量
        self.smap = []       # SMAP数据路径
        self.texture = []     # 土壤质地数据路径
        
        # label变量
        self.smap_unorm = []   # SMAP数据的非归一化值路径
        self.grid_ati = []    # 包含对应grid的ati
        self.insitu_ati = []  # 包含smap对应in-situ的ati 
        self.insitu_sm = []
        
        # 元信息
        self.valid_day_sequence = []  # 有效天数的完整序列
        
        if insitu_validation==False:
            ati_grid_root = os.path.join(root,'LABEL\ATI\GRID')
            print(ati_grid_root)
            subdir_list = sorted(os.listdir(ati_grid_root))

            # 遍历\LABEL\ATI\GRID下的所有子目录，即有效天数
            for subdir in subdir_list:
                subdir_path = os.path.join(ati_grid_root, subdir)
                if os.path.isdir(subdir_path):
                    self.valid_day_sequence.append(subdir)

            print('***************************Load data path******************************')
            print('valid day sequence:', self.valid_day_sequence)
            for day in self.valid_day_sequence: # 例如：2015187
                print('_______________________________' + str(day) + '_______________________________')

                # 获取今日所有有效的SMAPID
                ati_grid_paths = os.path.join(ati_grid_root, day)
                file_extension = '*.npy'
                ati_files = glob.glob(os.path.join(ati_grid_paths, file_extension))

                for ati_file in ati_files: # 例如：SMAPID=1
                    smapid = os.path.basename(ati_file).split('.')[0]
                    print('_____________________________smap cell: ' + str(smapid) + '_____________________________')

                    # 添加输入变量路径
                    self.smap.append(root + '\\INPUT\\SMAP\\' + str(day) + '\\' + str(smapid) + '.npy')
                    self.texture.append(root + '\\INPUT\\TEXTURE\\' + str(smapid) + '.npy')

                    # 显示添加的路径
                    print_path(root + '\\INPUT\\SMAP\\' + str(day) + '\\' + str(smapid) + '.npy')
                    print_path(root + '\\INPUT\\TEXTURE\\' + str(smapid) + '.npy')

                    # 添加LABEL变量路径
                    self.smap_unorm.append(root + '\\LABEL\\SMAP\\' + str(day) + '\\' + str(smapid) + '.npy')
                    self.grid_ati.append(root + '\\LABEL\\ATI\\GRID\\' + str(day) + '\\' + str(smapid) + '.npy')

                    # 显示添加的路径
                    print_path(root + '\\LABEL\\SMAP\\' + str(day) + '\\' + str(smapid) + '.npy')
                    print_path(root + '\\LABEL\\ATI\\GRID\\' + str(day) + '\\' + str(smapid) + '.npy')
        else:
            smap2insitu_root = os.path.join(root,'LABEL\SMAPID2INSITUID')
            subdir_list = sorted(os.listdir(smap2insitu_root))

            # 遍历\LABEL\SMAPID2INSITUID下的所有子目录，即有效天数
            for subdir in subdir_list:
                subdir_path = os.path.join(smap2insitu_root, subdir)
                if os.path.isdir(subdir_path):
                    self.valid_day_sequence.append(subdir)

            print('***************************Load data path******************************')
            for day in self.valid_day_sequence: # 例如：2015187
                print('_______________________________' + str(day) + '_______________________________')

                # 获取今日所有有效的SMAPID
                insitu_list_paths = os.path.join(smap2insitu_root, day)
                file_extension = '*.npy'
                insitu_files = glob.glob(os.path.join(insitu_list_paths, file_extension))

                for insitu_file in insitu_files: # 例如：SMAPID=1
                    smapid = os.path.basename(insitu_file).split('.')[0]
                    print('_____________________________smap cell: ' + str(smapid) + '_____________________________')

                    # 添加输入变量路径
                    self.smap.append(root + '\\INPUT\\SMAP\\' + str(day) + '\\' + str(smapid) + '.npy')
                    self.texture.append(root + '\\INPUT\\TEXTURE\\' + str(smapid) + '.npy')

                    # 显示添加的路径
                    print_path(root + '\\INPUT\\SMAP\\' + str(day) + '\\' + str(smapid) + '.npy')
                    print_path(root + '\\INPUT\\TEXTURE\\' + str(smapid) + '.npy')

                    # 添加LABEL变量路径
                    self.smap_unorm.append(root + '\\LABEL\\SMAP\\' + str(day) + '\\' + str(smapid) + '.npy')
                    
                    # 显示添加的路径
                    print_path(root + '\\LABEL\\SMAP\\' + str(day) + '\\' + str(smapid) + '.npy')
                    
                    # 一个 SMAP 对应多个 in-situ SM
                    smap_to_insitu = np.load(root + "\\LABEL\\SMAPID2INSITUID\\" + str(day) + '\\' + str(smapid) + '.npy')
                    insitu_sm_list = []
                    insitu_ati_list = []
                    for _id in smap_to_insitu:
                        # 将路径加入列表
                        insitu_sm_list.append(root + "\\LABEL\\SM\\" + str(day) + "\\" + str(_id) + ".npy")
                        insitu_ati_list.append(root + "\\LABEL\\ATI\\INSITU\\" + str(day) + "\\" + str(_id) + ".npy")

                        # 显示添加的路径
                        print_path(root + "\\LABEL\\SM\\" + str(day) + "\\" + str(_id) + ".npy")
                        print_path(root + "\\LABEL\\ATI\\INSITU\\" + str(day) + "\\" + str(_id) + ".npy")

                    # add the data of insitu in insitu_list
                    self.insitu_ati.append(insitu_ati_list)  
                    self.insitu_sm.append(insitu_sm_list)              
                      
    def __getitem__(self, idx):
        # 定义数据package
        data_pkg = {'processed_data': [], 'label_data': {}, 'meta_data': {}}
        
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
        
        # 选择堆叠为连接input特征的方式
        x = self.__stack__(smap, texture)
        data_pkg['processed_data'] = x
        
        # 加载label数据
        smap_unorm_path = self.smap_unorm[idx]
        smap_unorm = np.load(smap_unorm_path)
        data_pkg['label_data']['smap'] = smap_unorm
        
        if self.insitu_validation==False:
            ati_grid_path = self.grid_ati[idx]
            ati_grid = np.load(ati_grid_path)
            data_pkg['label_data']['ati_grid'] = ati_grid   
        else:
            # 一个SMAPID可能对应多个in-situ sm
            insitu_sm_path_list = self.insitu_sm[idx]
            insitu_ati_path_list = self.insitu_ati[idx]
            data_pkg['meta_data']['insituid'] = []
            data_pkg['label_data']['insitu_sm'] = []
            data_pkg['label_data']['insitu_ati'] = []
            for i in range(len(insitu_sm_path_list)):
                # 获取数据路径
                insitu_sm_path = insitu_sm_path_list[i]
                insitu_ati_path = insitu_ati_path_list[i]
                # 通过路径获取站点id
                insituid = re.findall(r'\d+', insitu_sm_path)[-1] #//2015091//1.npy
                data_pkg['meta_data']['insituid'].append(insituid)
                # 加载数据
                insitu_sm = np.load(insitu_sm_path, allow_pickle=True)
                insitu_ati = np.load(insitu_ati_path, allow_pickle=True)
                # 将一个站点所需的label数据list加入package
                data_pkg['label_data']['insitu_sm'].append(insitu_sm)
                data_pkg['label_data']['insitu_ati'].append(insitu_ati)
                
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
    
    def get_valid_day_sequence(self):
        return self.valid_day_sequence
    
def print_path(path):
    if os.path.exists(path):
        print("\033[32m" + path + "\033[0m")  # 绿色文本
    else:
        print("\033[31m" + path + "\033[0m")  # 红色文本 