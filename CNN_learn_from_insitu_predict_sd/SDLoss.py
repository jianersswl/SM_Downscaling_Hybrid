import torch
import torch.nn as nn

import matplotlib.pyplot as plt
import numpy as np

def display(pred_list, sm_pred_list, sm_insitu_list, flag):
    x = range(len(pred_list))
    # create a scatter plot
    # create a plot with three lines
    plt.plot(x, pred_list, label='sd_pred')
    plt.plot(x, sm_pred_list, label='sm_pred')
    plt.plot(x, sm_insitu_list, label='sm_insitu')

    # add a legend and axis labels
    plt.title('Result for {} Data'.format(flag))
    plt.legend()

    # display the plot
    plt.show()
    
def calculate_upscaling_sm(sm_bar, sm_bar_sd, ati, ati_bar, ati_bar_sd):
#     print('ati info:', (ati - ati_bar) / ati_bar_sd)
    return sm_bar + sm_bar_sd * (ati - ati_bar) / ati_bar_sd

def self_defined_loss(pred, label_data, batch_size, eds_lambda, flag):
    return physics_loss(pred, label_data, batch_size, flag), EDS_loss(pred, eds_lambda)

def physics_loss(pred, label_data, batch_size, flag):
    criterion = nn.MSELoss(reduction='mean')
    loss = 0
    pred_list = []
    sm_pred_list= []
    sm_insitu_list = []
    for i, y in enumerate(pred):      # each calculation in a batch
        # calculate the high resolution sm for each in situ
#         print(label_data[i])
        for situ_pkg in label_data[i]: # each calculation in a smap
            sm_situ = situ_pkg[0][0]
            sm_bar = situ_pkg[1][0]
            ati =  situ_pkg[2][0]
            atim = situ_pkg[2][1]
            atisd = situ_pkg[2][2]
            sm_pred = calculate_upscaling_sm(sm_bar, y, ati, atim, atisd)
#             _loss = criterion(sm_pred, torch.FloatTensor([sm_situ]))
            _loss = criterion(sm_pred, torch.tensor(sm_situ, dtype=torch.float32))
            loss = (loss + _loss)
            pred_list.append(y.item())
            sm_pred_list.append(sm_pred.item())
            sm_insitu_list.append(sm_situ)
    if flag=='Validing':
        print('sd_pred: {}'.format(pred))
        display(pred_list, sm_pred_list, sm_insitu_list, flag)
#     print('sm_pred: {}'.format(torch.FloatTensor(sm_pred_list)))
#     print('sm_insitu: {}'.format(torch.FloatTensor(sm_insitu_list)))
    if flag=='Testing':
        pass
    else:
        pass
    loss = loss / batch_size
    return loss

def EDS_loss(pred, penalty_lambda):
    pred = torch.unsqueeze(pred, dim=1)
    eds_sim_mat = torch.cdist(pred, pred)
    # 获取对角线上的值
    diag = torch.diag(eds_sim_mat)
    # 获取非对角线上的值
    non_dig_sum = torch.sum(eds_sim_mat - torch.diag(diag))
#     print('non_dig_sum:', non_dig_sum)
    # 计算均值
    non_dig_sum_mean = non_dig_sum/(eds_sim_mat.shape[0]*(eds_sim_mat.shape[0]-1))
#     print('non_dig_sum_mean：', non_dig_sum_mean)
    similarity = 1/(1+non_dig_sum_mean)
#     print('similarity:', similarity)
    loss = penalty_lambda*similarity*similarity
    return loss
    