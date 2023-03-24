# I/O
import os
import numpy as np
# Pytorch
from torch.utils.data import Dataset

class SMAPDataset(Dataset):
    '''
    root: root of input data
    temperal_sequence: the sequence of valid day
    spatial_sequence: the sequence of valid SMAPID
    test: the flag to identify if it is the test dataset
    '''
    def __init__(self, root, sequence):
        # variables for input
        self.smap = []
        self.texture = []
        
        # variables for output
        self.sm = []
        self.smap_unorm = []
        self.ati = [] # contains [ati, atim, atisd] in each element
        
        print('***************************Load data path******************************')
        for i in sequence.keys(): # for example: 2015187
            print('_______________________________' + str(i) + '_______________________________')
            for j in sequence[i]: # for example: 1
                print('_____________________________smap cell: ' + str(j) + '_____________________________')
                # add path for input variables
                self.smap.append(root + 'INPUT\\SMAP\\' + i + '\\' + str(j) + '.npy')
                self.texture.append(root + 'INPUT\\TEXTURE\\' + str(j) + '.npy')
                # display adding path
                print((root + 'INPUT\\SMAP\\' + i + '\\' + str(j) + '.npy'))
                print((root + 'INPUT\\TEXTURE\\' + str(j) + '.npy'))
                print(os.path.exists(root + 'INPUT\\SMAP\\' + i + '\\' + str(j) + '.npy'))
                print(os.path.exists(root + 'INPUT\\TEXTURE\\' + str(j) + '.npy'))
                      
                # one smap to many in-situ sm
                self.smap_unorm.append(root + 'LABEL\\SMAP\\' + i + '\\' + str(j) + '.npy')
                smap_to_insitu = np.load(root + "LABEL\\SMAPID2INSITUID\\" + i + '\\' + str(j) + '.npy')
                insitu_sm_list = []
                insitu_ati_list = []
                for _id in smap_to_insitu:
                    insitu_sm_list.append(root + "LABEL\\SM\\" + i + "\\" + str(_id) + ".npy")
                    insitu_ati_list.append(root + "LABEL\\ATI\\" + i + "\\" + str(_id) + ".npy")
                    # display adding path
                    print((root + "LABEL\\SM\\" + i + "\\" + str(_id) + ".npy"))
                    print((root + "LABEL\\ATI\\" + i + "\\" + str(_id) + ".npy"))
                    print(os.path.exists(root + "LABEL\\SM\\" + i + "\\" + str(_id) + ".npy"))
                    print(os.path.exists(root + "LABEL\\ATI\\" + i + "\\" + str(_id) + ".npy"))
                      
                # add the data of insitu in insitu_list
                self.sm.append(insitu_sm_list)
                self.ati.append(insitu_ati_list)    
                      
    def __getitem__(self, idx):
        smap_path = self.smap[idx]
        smap_unorm_path = self.smap_unorm[idx]
        texture_path = self.texture[idx]
        smap = np.load(smap_path)
        smap_unorm = np.load(smap_unorm_path)
        texture = np.load(texture_path)
        
        data_pkg = {'processed_data': [], 'label_data': []}
        
        # choose flatten as the way to concatenate the input feature
        x = self.__flatten__(smap, texture)
        data_pkg['processed_data'] = x
        
        sm_list = self.sm[idx]
        ati_list = self.ati[idx]
        y = [] # y -> [[sm, smap, ati], ...], sm -> [float], smap -> [float],1 ati -> [ati, atim, atisd]
        for i in range(len(sm_list)):
            sm_path = sm_list[i]
            ati_path = ati_list[i]
            sm = np.load(sm_path, allow_pickle=True)
            ati = np.load(ati_path, allow_pickle=True)
            data_pkg['label_data'].append([sm, smap_unorm, ati])      # other_data -> [[sm, smap, ati], ...], 
                                                                    # sm -> [float]
                                                                    # smap -> [float], 
                                                                    # ati -> [ati, atim, atisd]
            
            # test: each element in list of batch should be of equal size
#             break
        return data_pkg

    def __len__(self):
        return len(self.smap)

    ### the way to concatenate input data
    def __flatten__(self, smap, texture):
        # normalization is done before loading
        texture_flat = texture.flatten()
        return  np.concatenate((smap, texture_flat), axis=0)
    
    def get_input_shape(self, idx):
        data_pkg = self.__getitem__(0)
        return data_pkg['processed_data'].shape
    