# Numerical Operations
import numpy as np
import torch

def random_spatial_sequence(split_rate, full_sequence):
    counts = len(full_sequence)
    len1 = int(counts * split_rate)
    len2 = counts - len1
    index1 = np.random.choice(full_sequence, len1, replace=False)
    index2 = np.setdiff1d(full_sequence, index1)
    print('**************************Data Spliting***************************')
    print('Spliting Rate: ', split_rate)
    print(len1, 'of Dataset1: ',index1)
    print(len2, 'of Dataset2: ',index2)
    print('**************************Data Spliting***************************')
    return index1, index2

def collate_insitu_validation(batch):
    processed_data = torch.FloatTensor(np.asarray([sample['processed_data'] for sample in batch]))
    
    label_data = {}
    _label_data = [sample['label_data'] for sample in batch]
    label_data['smap'] = torch.FloatTensor(np.asarray([sample['smap'] for sample in _label_data]))
    label_data['insitu_sm'] = [sample['insitu_sm'] for sample in _label_data]
    label_data['insitu_ati'] = [sample['insitu_ati'] for sample in _label_data]
    
    meta_data = [sample['meta_data'] for sample in batch]
    
    # 将所有需要传递的数据都保存在同一个字典中
    batch_dict = {'processed_data': processed_data,
                'label_data': label_data,
                 'meta_data': meta_data}
    
    return batch_dict

def collate_fn(batch):
    # 从每个样本的字典中获取处理结果、标签和其他数据，并将它们存储在同一个字典中
    processed_data = torch.FloatTensor(np.asarray([sample['processed_data'] for sample in batch]))
    
    label_data = {}
    _label_data = [sample['label_data'] for sample in batch]
    label_data['smap'] = torch.FloatTensor(np.asarray([sample['smap'] for sample in _label_data]))
    label_data['ati_grid'] = torch.FloatTensor(np.asarray([sample['ati_grid'] for sample in _label_data]))
    
    meta_data = [sample['meta_data'] for sample in batch]
    
    # 将所有需要传递的数据都保存在同一个字典中
    batch_dict = {'processed_data': processed_data,
                'label_data': label_data,
                 'meta_data': meta_data}
    
    return batch_dict